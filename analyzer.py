import os
import re
import json
import glob
import logging
import nltk
import spacy
import string
import pdfplumber
import pytesseract
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image, ImageEnhance
import io
import time
from typing import List, Dict, Optional
from nltk.corpus import stopwords
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI

# logging configuration
class MultiLevelLogger:
    def __init__(self, log_file: str):
        # Console handler - INFO and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        # File handler - DEBUG and above, with more details
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

# Download NLTK stopwords if not already present
nltk.download('stopwords', quiet=True)

class PDFProcessor:
    """
    Handles loading and extracting text from PDFs, segmenting into sentences, preserving original text.
    Optimized for speed using batched processing and reduced redundant operations.
    """

    def __init__(self, enable_ocr: bool = False, num_workers: int = None):
        self.num_workers = num_workers or os.cpu_count()
        self.enable_ocr = enable_ocr
        self.nlp = spacy.load('en_core_web_sm')  # For sentence segmentation
        # Set Tesseract path for OCR if on Windows
        # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Ensure this path is correct

    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        return ' '.join(text.split())

    def process_page(self, page) -> List[Dict[str, str]]:
        extracted_content = []
        try:
            text = page.extract_text()
            if text:
                logging.debug(f"Raw extracted text (snippet): {text[:200]}...")
                text = self.clean_text(text)
                if text.strip():
                    doc = self.nlp(text)
                    for sent in doc.sents:
                        extracted_content.append({'original': sent.text})

            if self.enable_ocr:
                for image in page.images:
                    try:
                        # Convert image data to PIL Image more robustly
                        img_data = image['stream'].get_data()
                        try:
                            img = Image.frombytes(
                                mode='RGB',
                                size=(int(image['width']), int(image['height'])),
                                data=img_data
                            )
                        except:
                            # Fallback to opening as bytes
                            img = Image.open(io.BytesIO(img_data))

                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        # Improve OCR accuracy with preprocessing
                        img = ImageEnhance.Contrast(img).enhance(2.0)
                        img = ImageEnhance.Sharpness(img).enhance(2.0)

                        # OCR with improved config
                        image_text = pytesseract.image_to_string(
                            img,
                            config='--psm 3 --oem 3'
                        )

                        image_text = self.clean_text(image_text)
                        if image_text.strip():
                            img_doc = self.nlp(image_text)
                            for sent in img_doc.sents:
                                if len(sent.text.split()) > 2:  # Only add if meaningful
                                    extracted_content.append({'original': sent.text})
                    except Exception as e:
                        logging.warning(f"Image processing error: {str(e)}")
                        continue

        except Exception as e:
            logging.error(f"Page processing error: {str(e)}")

        return extracted_content

    def process_single_pdf(self, pdf_path: str) -> Dict[str, List[Dict[str, str]]]:
        extracted_content = {'sentences': [], 'source': pdf_path}
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    extracted_content['sentences'].extend(self.process_page(page))
        except Exception as e:
            logging.error(f"Error processing PDF {pdf_path}: {e}")
            return None
        return extracted_content

    def process_pdfs(self, pdf_paths: List[str]) -> List[Dict[str, List[Dict[str, str]]]]:
        logging.info(f"Processing {len(pdf_paths)} PDFs using {self.num_workers} workers (OCR={self.enable_ocr})")
        results = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for result in tqdm(executor.map(self.process_single_pdf, pdf_paths),
                               total=len(pdf_paths), desc="Processing PDFs"):
                if result is not None:
                    results.append(result)
        return results


class TextCleaner:
    """
    Applies cleaning to individual sentences.
    Available lemmatization methods:
    - 'spacy': Uses SpaCy's lemmatizer to convert words to their base form (e.g., 'running' -> 'run')
    - 'none': No lemmatization, keeps original word forms
    """
    def __init__(self, keep_dates: bool = False, keep_numbers: bool = False, lemmatization_method: str = 'spacy'):
        """
        Initialize TextCleaner with configurable options.
        
        Args:
            keep_dates: If True, preserves dates in text. If False, removes them.
            keep_numbers: If True, preserves numbers in text. If False, removes them.
            lemmatization_method: Either 'spacy' or 'none':
                - 'spacy': Converts words to base form (e.g., 'running' -> 'run')
                - 'none': Keeps original word forms
        """
        self.nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"])
        self.stopwords = set(stopwords.words('english'))
        self.url_pattern = re.compile(r"(https?://\S+)|(\bwww.\S+\b)", re.IGNORECASE)
        self.date_pattern = re.compile(r"\b(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}|\d{4})\b")
        self.citation_pattern = re.compile(r"[\[\(]?\d+[\]\)]?|(\d+)")
        self.numeric_ref_pattern = re.compile(r"\b\d+\b")
        self.keep_dates = keep_dates
        self.keep_numbers = keep_numbers
        self.lemmatization_method = lemmatization_method

    def remove_noise(self, text: str) -> str:
        text = self.url_pattern.sub(" ", text)
        text = self.citation_pattern.sub(" ", text)
        if self.keep_dates:
            text = self.date_pattern.sub(" ", text)
        if self.keep_numbers:
            text = self.numeric_ref_pattern.sub(" ", text)
        text = ' '.join(text.split())
        return text

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        if self.lemmatization_method == 'spacy':
            doc = self.nlp(text.lower())
            tokens = [
                token.lemma_
                for token in doc
                if not token.is_stop
                and not token.is_punct
                and not (not self.keep_numbers and token.like_num)
                and len(token.lemma_) >= 2
                and not all(ch in string.punctuation for ch in token.lemma_)
            ]
        elif self.lemmatization_method == 'none':
            # Just tokenize without lemmatization
            doc = self.nlp(text.lower())
            tokens = [
                token.text
                for token in doc
                if not token.is_stop
                and not token.is_punct
                and not (not self.keep_numbers and token.like_num)
                and len(token.text) >= 2
                and not all(ch in string.punctuation for ch in token.text)
            ]
        else:
            raise ValueError(f"Unknown lemmatization method: {self.lemmatization_method}")
        
        return tokens

    def clean_sentence(self, text: str) -> str:
        logging.debug(f"Pre-cleaning text (snippet): {text[:200]}...")
        text = self.remove_noise(text)
        tokens = self.tokenize_and_lemmatize(text)
        cleaned = ' '.join(tokens)
        logging.debug(f"Post-cleaning text (snippet): {cleaned[:200]}...")
        return cleaned


class ThematicAnalyzer:
    """
    Performs semantic analysis, including both supervised and unsupervised modes.
    In supervised mode, matches sentences to key concepts by similarity.
    In unsupervised mode, clusters sentences using KMeans, extracts keywords,
    and obtains short theme names from GPT.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        embedding_model_name: str = 'all-mpnet-base-v2',
        similarity_threshold: float = 0.7,
        n_clusters_for_unsupervised_codes: int = 10,
        n_clusters_for_themes: int = 10,
        text_config: Dict = None,
        research_question: Optional[str] = None
    ):
        self.openai_api_key = openai_api_key
        self.research_question = research_question
        self.n_clusters_for_unsupervised_codes = n_clusters_for_unsupervised_codes
        self.n_clusters_for_themes = n_clusters_for_themes
        self.similarity_threshold = similarity_threshold

        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)

        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Initialize TextCleaner with config
        text_config = text_config or {}
        self.text_cleaner = TextCleaner(
            keep_dates=text_config.get('keep_dates', False),
            keep_numbers=text_config.get('keep_numbers', False),
            lemmatization_method=text_config.get('lemmatization_method', 'spacy')
        )

    ############################################################################
    # 1) SUPERVISED: Identify codes from sentences given key_concepts
    ############################################################################
    def identify_codes_from_sentences(
        self,
        sentences: List[Dict[str, str]],
        key_concepts: Optional[List[str]] = None
    ) -> Dict[str, List[int]]:
        """
        Identifies codes within sentences in supervised mode
        based on semantic similarity with provided key_concepts.
        """
        if not sentences:
            logging.warning("No sentences provided for code identification.")
            return {}

        if not key_concepts:
            # If no key_concepts are provided, we skip supervised coding entirely.
            return {}

        cleaned_sentences = [
            self.text_cleaner.clean_sentence(s['original']) for s in sentences
        ]
        sentence_embeddings = self.embedding_model.encode(cleaned_sentences)
        concept_embeddings = self.embedding_model.encode(key_concepts)
        similarity_matrix = cosine_similarity(sentence_embeddings, concept_embeddings)

        sentence_codes = {}
        match_found = False

        for i, sentence in enumerate(sentences):
            for j, concept in enumerate(key_concepts):
                if similarity_matrix[i, j] > self.similarity_threshold:
                    match_found = True
                    if concept not in sentence_codes:
                        sentence_codes[concept] = []
                    sentence_codes[concept].append(i)

        if not match_found:
            logging.warning("No codes identified in supervised mode.")

        return sentence_codes

    ############################################################################
    # 2) UNSUPERVISED CLUSTERING: KMeans on all sentences
    ############################################################################
    def cluster_sentences_unsupervised(
        self,
        sentences: List[str],
        n_clusters: int
    ) -> np.ndarray:
        """
        Perform KMeans clustering on sentence embeddings to group them into `n_clusters`.
        Returns an array of cluster labels, one per sentence.
        """
        logging.info(f"Performing unsupervised clustering into {n_clusters} clusters...")
        sentence_embeddings = self.embedding_model.encode(sentences)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(sentence_embeddings)
        return labels

    ############################################################################
    # 3) Extract Keywords for Each Cluster
    ############################################################################
    def extract_cluster_keywords(
        self,
        sentences: List[str],
        cluster_labels: np.ndarray,
        top_n: int = 5
    ) -> Dict[int, List[str]]:
        """
        For each cluster, extract top keywords using a TF-IDF approach.
        Returns a dict: cluster_label -> list of top keywords
        """
        logging.info("Extracting keywords for each cluster...")
        df = pd.DataFrame({'text': sentences, 'cluster': cluster_labels})

        # Vectorize text
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000
        )
        tfidf_matrix = vectorizer.fit_transform(df['text'])
        feature_names = vectorizer.get_feature_names_out()

        cluster_keywords = {}
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_text_indices = df.index[df['cluster'] == cluster_id].tolist()
            if not cluster_text_indices:
                cluster_keywords[cluster_id] = ["none"]
                continue

            # Aggregate rows for this cluster
            cluster_tfidf = tfidf_matrix[cluster_text_indices, :]
            # Average the TF-IDF scores
            mean_tfidf = cluster_tfidf.mean(axis=0).A1
            top_indices = mean_tfidf.argsort()[::-1][:top_n]
            top_feats = [feature_names[idx] for idx in top_indices]
            cluster_keywords[cluster_id] = top_feats

        return cluster_keywords

    ############################################################################
    # 4) GPT Prompting: Decide if cluster is "skip" or name
    ############################################################################
    def gpt_determine_theme_name(
        self,
        keywords: List[str]
    ) -> str:
        """
        Pass the extracted keywords to GPT to determine if cluster is meaningful or should be skipped.
        MUST only output 'skip' or a short name.
        """
        if not self.openai_api_key:
            # If no GPT, just fallback to generic name or skip
            return "skip"

        prompt_text = (
            "You are an advanced theme identification assistant. "
            "We have these keywords extracted from a cluster of sentences:\n"
            f"{', '.join(keywords)}\n\n"
            "If these keywords represent a meaningful theme for qualitative analysis and if they related to the research, reply with a short (2-4 words) but VERY, VERY specific theme name.\n"
            "If the majority of keywords is not in any way relevant to the research question or if they represent document metadata, references or similar, you MUST ALWAYS reply with 'skip'.\n"
            f"The research question is: {self.research_question}\n"
            "Your output MUST be exactly one line, containing EITHER 'skip' OR the short theme name. Nothing else."
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  
                messages=[ 
                    {
                        "role": "system",
                        "content": (
                            "You are an expert text classification model. "
                            "You strictly output either 'skip' or a short theme name; no additional text."
                        )
                    },
                    {"role": "user", "content": prompt_text},
                ],
                max_tokens=20,
                temperature=0.5
            )
            gpt_output = response.choices[0].message.content.strip()
            logging.debug(f"GPT naming output: {gpt_output}")
            gpt_output = re.sub(r'^[\"\']|[\"\']$', '', gpt_output).strip()

            if "\n" in gpt_output:
                lines = gpt_output.split("\n")
                gpt_output = lines[0].strip()

            if not gpt_output:
                return "skip"
            if len(gpt_output.split()) > 15:
                return "skip"
            return gpt_output

        except Exception as e:
            logging.error(f"Error calling GPT for cluster naming: {e}")
            return "skip"

    ############################################################################
    # 5) Summarize Theme with GPT
    ############################################################################
    def gpt_summarize_theme(
        self,
        theme_name: str,
        example_sentences: List[str],
        keywords: List[str],
        max_sentences: int = 5
    ) -> str:
        """
        Provide a short GPT-based summary of the theme using example sentences and keywords.
        """
        if not self.openai_api_key:
            return "No GPT summary (missing API key)."

        truncated = example_sentences[:max_sentences]
        text_block = " | ".join(truncated)
        keywords_text = ", ".join(keywords)

        prompt_text = (
            f"Theme: {theme_name}\n"
            f"Keywords defining this theme: {keywords_text}\n"
            "Example text snippets:\n"
            f"{text_block}\n"
            f"Based on both the keywords and example snippets, provide a concise summary (1-2 sentences) "
            f"of this theme that emerged from the thematic analysis for this research question: {self.research_question}"
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are an expert in thematic analysis. "
                            "Synthesize both keywords and examples to create precise theme summaries."
                        )
                    },
                    {"role": "user", "content": prompt_text},
                ],
                max_tokens=100,
                temperature=0.2
            )
            summary_text = response.choices[0].message.content.strip()
            summary_text = re.sub(r'[\r\n]+', ' ', summary_text).strip()
            return summary_text

        except Exception as e:
            logging.error(f"GPT summarization error: {e}")
            return "Summary unavailable."

    ############################################################################
    # 6) Summaries and Conclusion 
    ############################################################################

    def gpt_generate_overall_conclusion(
        self,
        themes_df: pd.DataFrame,
        research_question: str
    ) -> str:
        """
        Generate comprehensive conclusion using theme counts, summaries, and research question.
        """
        if not self.openai_api_key:
            return "No GPT conclusion available (no API key)."

        if themes_df.empty:
            logging.warning("Empty DataFrame provided to generate conclusion.")
            return "No data available for conclusion generation."

        try:
            # Get top themes with their counts and summaries
            top_themes = themes_df.head(5)
            
            # Create structured context for GPT
            themes_context = []
            for _, row in top_themes.iterrows():
                theme_info = {
                    "name": row["Theme"],
                    "count": row["Count"],
                    "summary": row["GPT_Theme_Summary"],
                    "keywords": row["Keywords"]
                }
                themes_context.append(theme_info)
            
            # Create prompt with rich context
            theme_details = []
            for t in themes_context:
                detail = (
                    f"Theme: {t['name']}\n"
                    f"Frequency: {t['count']} occurrences\n"
                    f"Keywords: {t['keywords']}\n"
                    f"Summary: {t['summary']}"
                )
                theme_details.append(detail)
            
            themes_text = "\n\n".join(theme_details)
            
            prompt_text = (
                f"Based on a thematic analysis of documents, here are the main themes discovered:\n\n"
                f"{themes_text}\n\n"
                f"Research Question: {research_question}\n\n"
                "Provide a concise, evidence-based conclusion (2-3 sentences) that:\n"
                "1. Identifies the dominant themes based on their frequency\n"
                "2. Synthesizes their key findings\n"
                "3. Directly answers the research question"
            )

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert in qualitative data analysis who provides "
                                "authoritative conclusions based on thematic analysis results."
                            )
                        },
                        {"role": "user", "content": prompt_text}
                    ],
                    max_tokens=150,
                    temperature=0.2
                )
                conclusion = response.choices[0].message.content.strip()
                return conclusion if conclusion else "No meaningful conclusion generated."

            except Exception as e:
                logging.error(f"Error in GPT API call: {e}")
                return "Error generating conclusion via GPT."

        except Exception as e:
            logging.error(f"Error in conclusion generation: {e}")
            return "Error analyzing themes."


def run_analysis(
    pdf_directory: str,
    output_path: str,
    openai_api_key: str,
    key_concepts: Optional[List[str]],
    suffix: str,
    keep_dates: bool,
    keep_numbers: bool,
    lemmatization_method: str,
    research_question: Optional[str] = None
):
    # Create output directory
    output_path_suffix = os.path.join(output_path, suffix)
    os.makedirs(output_path_suffix, exist_ok=True)

    # 1. PDF extraction
    pdf_paths = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    pdf_processor = PDFProcessor(enable_ocr=True)
    processed_results = pdf_processor.process_pdfs(pdf_paths)

    # 2. Organize sentences
    documents_sentences = {}
    all_sentences_list = []  
    for result in processed_results:
        doc_path = result['source']
        documents_sentences[doc_path] = result['sentences']
        for s in result['sentences']:
            all_sentences_list.append((doc_path, s['original']))

    # 3. Initialize ThematicAnalyzer
    analyzer = ThematicAnalyzer(
        openai_api_key=openai_api_key,
        n_clusters_for_unsupervised_codes=10,
        n_clusters_for_themes=10,
        text_config={
            'keep_dates': keep_dates,
            'keep_numbers': keep_numbers,
            'lemmatization_method': lemmatization_method
        },
        research_question=research_question
    )

    if key_concepts:
        logging.info(f"Running SUPERVISED scenario with {len(key_concepts)} key concepts...")

        # 1) Clean all sentences
        cleaned_sentences_list = []
        for (doc_path, text) in all_sentences_list:
            cleaned = analyzer.text_cleaner.clean_sentence(text)
            cleaned_sentences_list.append(cleaned if cleaned.strip() else "EMPTY")

        # 2) Compute similarity scores
        sentence_embeddings = analyzer.embedding_model.encode(cleaned_sentences_list)
        concept_embeddings = analyzer.embedding_model.encode(key_concepts)
        similarity_matrix = cosine_similarity(sentence_embeddings, concept_embeddings)

        # 3) For each concept, gather relevant sentences
        theme_info = {}
        for concept_idx, concept in enumerate(key_concepts):
            matching_indexes = np.where(similarity_matrix[:, concept_idx] > analyzer.similarity_threshold)[0]
            if len(matching_indexes) == 0:
                continue

            theme_sentences = []
            docset = set()
            for idx in matching_indexes:
                docset.add(os.path.basename(all_sentences_list[idx][0]))
                theme_sentences.append(all_sentences_list[idx][1])

            keywords = analyzer.extract_cluster_keywords(
                sentences=[cleaned_sentences_list[i] for i in matching_indexes],
                cluster_labels=np.zeros(len(matching_indexes)),  # Single cluster assumption
                top_n=5
            ).get(0, ["none"])

            summary_txt = analyzer.gpt_summarize_theme(
                concept, 
                theme_sentences[:5],
                keywords,  # Pass keywords to summary generation
                max_sentences=3
            )

            theme_info[concept] = {
                "sentences": theme_sentences,
                "docs": docset,
                "count": len(theme_sentences),
                "keywords": keywords,
                "summary": summary_txt
            }

        final_themes = []
        for theme_name, info in theme_info.items():
            row = {
                "Theme": theme_name,
                "Keywords": ", ".join(info["keywords"]),
                "GPT_Theme_Summary": info["summary"],
                "Example_Sentences": " | ".join(info["sentences"][:3]),
                "Documents": ", ".join(sorted(info["docs"])),
                "Count": info["count"]
            }
            final_themes.append(row)

        themes_df = pd.DataFrame(final_themes)
        if not themes_df.empty:
            themes_df.sort_values("Count", ascending=False, inplace=True)

            overall_conclusion = analyzer.gpt_generate_overall_conclusion(
                themes_df,
                research_question
            )

            csv_output_path = os.path.join(output_path_suffix, f"supervised_thematic_summary_{suffix}.csv")
            with open(csv_output_path, "w", encoding="utf-8") as csv_file:
                if overall_conclusion:
                    csv_file.write(f"Overall_Conclusion,\"{overall_conclusion}\"\n\n")
                themes_df.to_csv(csv_file, index=False)

            plt.figure(figsize=(10, 5))
            sns.barplot(x="Theme", y="Count", data=themes_df)
            plt.xticks(rotation=45, ha='right')
            plt.title(f"Supervised Themes by Match Count ({suffix})")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path_suffix, f"supervised_theme_distribution_{suffix}.png"))
            plt.close()

            logging.info(f"Supervised analysis complete. Results in {csv_output_path}.")
        else:
            logging.error("No valid themes found in supervised analysis.")
        return  # End execution after supervised flow

    else:
        logging.info("Running UNSUPERVISED scenario (no key concepts provided).")

        # 1) Clean all sentences
        cleaned_sentences_list = []
        for (doc_path, text) in all_sentences_list:
            cleaned = analyzer.text_cleaner.clean_sentence(text)
            cleaned_sentences_list.append(cleaned if cleaned.strip() else "EMPTY")

        # 2) Cluster with KMeans
        cluster_labels = analyzer.cluster_sentences_unsupervised(
            sentences=cleaned_sentences_list,
            n_clusters=analyzer.n_clusters_for_unsupervised_codes
        )

        # 3) Extract top keywords per cluster
        keywords_per_cluster = analyzer.extract_cluster_keywords(
            sentences=cleaned_sentences_list,
            cluster_labels=cluster_labels,
            top_n=5
        )

        # 4) Determine themes for each cluster and gather info
        cluster_themes = {}
        cluster_info = {}
        for cluster_id in sorted(np.unique(cluster_labels)):
            kw = keywords_per_cluster[cluster_id]
            theme_name_or_skip = analyzer.gpt_determine_theme_name(kw)

            cluster_sentences = []
            docset = set()
            indexes_in_cluster = np.where(cluster_labels == cluster_id)[0]
            for idx in indexes_in_cluster:
                docset.add(os.path.basename(all_sentences_list[idx][0]))
                cluster_sentences.append(all_sentences_list[idx][1])

            cluster_info[cluster_id] = {
                "theme": theme_name_or_skip,
                "sentences": cluster_sentences,
                "docs": docset,
                "count": len(cluster_sentences),
                "keywords": kw
            }

            if theme_name_or_skip.lower() != "skip":
                cluster_themes[cluster_id] = theme_name_or_skip

        final_themes = []
        for cluster_id, theme_name in cluster_themes.items():
            example_sents = cluster_info[cluster_id]["sentences"][:5]
            summary_txt = analyzer.gpt_summarize_theme(
                theme_name,
                example_sents,
                cluster_info[cluster_id]["keywords"],  # Pass cluster keywords
                max_sentences=3
            )
            row = {
                "Theme": theme_name,
                "Keywords": ", ".join(cluster_info[cluster_id]["keywords"]),
                "GPT_Theme_Summary": summary_txt,
                "Example_Sentences": " | ".join(example_sents[:3]),
                "Documents": ", ".join(sorted(cluster_info[cluster_id]["docs"])),
                "Count": cluster_info[cluster_id]["count"]
            }
            final_themes.append(row)

        themes_df = pd.DataFrame(final_themes)
        themes_df.sort_values("Count", ascending=False, inplace=True)

        csv_output_path = os.path.join(output_path_suffix, f"unsupervised_thematic_summary_{suffix}.csv")
        
        # Generate enhanced overall conclusion
        overall_conclusion = ""
        if not themes_df.empty:
            overall_conclusion = analyzer.gpt_generate_overall_conclusion(
                themes_df,
                research_question
            )

        with open(csv_output_path, "w", encoding="utf-8") as csv_file:
            if overall_conclusion:
                csv_file.write(f"Overall_Conclusion,\"{overall_conclusion}\"\n\n")
            themes_df.to_csv(csv_file, index=False)

        logging.info(f"Unsupervised clustering complete. Results in {csv_output_path}.")

        if not themes_df.empty:
            plt.figure(figsize=(10, 5))
            sns.barplot(x="Theme", y="Count", data=themes_df)
            plt.xticks(rotation=45, ha='right')
            plt.title(f"Unsupervised Themes by Cluster Count ({suffix})")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path_suffix, f"unsupervised_theme_distribution_{suffix}.png"))
            plt.close()


def main():
    # Initialize logging
    log_file = os.path.join("output", "analysis_log.txt")
    os.makedirs("output", exist_ok=True)
    MultiLevelLogger(log_file)

    # CONFIGURATION
    pdf_directory = "pdfs"
    output_path = "output"
    os.makedirs(output_path, exist_ok=True)
    openai_api_key = None  # Replace with your actual API key

    research_question = None # Insert your research question here

    text_cleaning_config = {
        "keep_dates": True,
        "keep_numbers": True,
        "lemmatization_method": "none"
    }

    key_concepts = [
        # Insert your key concepts here, if any.
    ]

    if key_concepts:
        logging.info(f"Using {len(key_concepts)} key concepts for supervised analysis.")
        run_analysis(
            pdf_directory, 
            output_path, 
            openai_api_key, 
            key_concepts=key_concepts, 
            suffix="with_keywords",
            research_question=research_question,
            **text_cleaning_config
        )
    else:
        logging.info("No key concepts provided. Running unsupervised analysis.")
        run_analysis(
            pdf_directory, 
            output_path, 
            openai_api_key, 
            key_concepts=None, 
            suffix="unsupervised",
            research_question=research_question,
            **text_cleaning_config
        )

if __name__ == "__main__":
    main()
