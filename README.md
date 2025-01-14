# Thematic Analysis Tool

A Python-based tool for performing thematic analysis on collections of PDF documents. It supports both supervised (using key concepts) and unsupervised (clustering) analysis, leveraging NLP techniques and GPT models for insightful results.

This thematic analyzer can also be run directly in this [google collab notebook](https://colab.research.google.com/drive/181m_di0C8bzOgJxSKUgpuklvSpTzEQOV?usp=sharing). To do your thematic analysis, follow the instructions in the notebook. No additional setup is required.

## Features

-   **PDF Processing:** Extracts text from PDFs, including handling scanned documents with OCR (optional).
-   **Text Cleaning:** Customizable cleaning options to remove noise, handle dates, numbers, and choose lemmatization methods.
-   **Supervised Analysis:** Identify themes based on provided key concepts using semantic similarity.
-   **Unsupervised Analysis:** Cluster sentences using KMeans, extract keywords, and generate theme names using GPT.
-   **GPT Integration:** Leverages GPT models for theme naming, summarization, and generating overall conclusions (optional, requires OpenAI API key).
-   **Output:** Generates CSV summaries of themes, including example sentences, associated documents, and GPT-generated summaries. Also produces visualizations of theme distributions.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/YOUR_USERNAME/thematic-analysis-tool.git
    cd thematic-analysis-tool
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Install Tesseract OCR (Optional):**

    -   If you want to use OCR for scanned PDFs, install Tesseract OCR:
        -   **Windows:** Download from the [official website](https://tesseract-ocr.github.io/tessdoc/Home.html) and add the installation directory to your system's PATH. You may also need to set the `pytesseract.pytesseract.tesseract_cmd` variable in your script to point to the `tesseract.exe` file.
        -   **macOS:** `brew install tesseract`
        -   **Linux:** `sudo apt-get install tesseract-ocr` (or equivalent for your distribution)
    -   Enable OCR in the script by setting the `enable_ocr` flag in the `PDFProcessor` class.

4. **OpenAI API Key (Optional):**

    -   If you want to use the GPT features (theme naming, summarization, conclusion generation), you'll need an OpenAI API key.
    -   Set the `OPENAI_API_KEY` environment variable:
        -   **Linux/macOS:**

            ```bash
            export OPENAI_API_KEY="your_api_key_here"
            ```

        -   **Windows:**

            ```bash
            setx OPENAI_API_KEY "your_api_key_here"
            ```

## Usage

1. **Prepare your PDFs:** Place the PDF files you want to analyze in the `pdfs` directory (or change the `pdf_directory` variable in the script).
2. **Configure the script:**
    *   Open the Python script in a text editor.
    *   **Modify the following variables at the beginning of the `main()` function according to your needs:**
        *   `pdf_directory`: Path to the directory containing your PDF files.
        *   `output_path`: Path to the directory where you want to store the output.
        *   `openai_api_key`: If you have an OpenAI API key and want to use GPT features, set this variable using `os.environ.get("OPENAI_API_KEY")` and follow the instruction above to set your API key as an evironment variable. Otherwise leave as `None`.
        *   `research_question`: **Enter your research question here.**
        *   `key_concepts`: If you want to perform supervised analysis, add your key concepts to this list. Otherwise, leave it empty for unsupervised analysis.
        *   `text_cleaning_config`: Adjust the text cleaning parameters (`keep_dates`, `keep_numbers`, `lemmatization_method`) as needed.
3. **Run the script:**

    ```bash
    python thematic_analysis.py
    ```

## Output

The script will generate the following outputs in the specified output directory:

-   `analysis_log.txt`: A log file containing details of the analysis process.
-   `supervised_thematic_summary_<suffix>.csv`: Summary of themes identified in supervised mode (if key concepts were provided).
-   `unsupervised_thematic_summary_<suffix>.csv`: Summary of themes identified in unsupervised mode.
-   `supervised_theme_distribution_<suffix>.png`: Visualization of theme distribution in supervised mode.
-   `unsupervised_theme_distribution_<suffix>.png`: Visualization of theme distribution in unsupervised mode.

The CSV files contain the following columns:

-   `Theme`: The name of the theme.
-   `Keywords`: Top keywords associated with the theme.
-   `GPT_Theme_Summary`: A GPT-generated summary of the theme (if OpenAI API key is provided).
-   `Example_Sentences`: Example sentences belonging to the theme.
-   `Documents`: List of documents where the theme was found.
-   `Count`: The number of sentences associated with the theme.
-   `Overall_Conclusion`: A GPT-generated overall conclusion based on the analysis (if OpenAI API key is provided).

## License

This project is licensed under the MIT License

## Contributing

Contributions are welcome!
