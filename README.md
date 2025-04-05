# Multilingual Syntax Analyzer

This application allows you to analyze text in multiple languages to generate syntax trees, sentence diagrams, and understand grammatical rules.

## Features

* **Multiple Language Support** : Analyze text in 12 different languages including English, Spanish, French, German, Chinese, and more.
* **Dependency Parsing** : Visualize the dependency relationships between words in a sentence.
* **Constituency Parsing** : View the hierarchical structure of sentences.
* **Grammatical Rule Analysis** : Extract and display grammatical relationships and patterns.
* **File Upload** : Process text from uploaded documents.

## Setup Instructions

1. **Install dependencies** :

```
   pip install -r requirements.txt
```

1. **Download language models** :
   The application will automatically download required language models on first use, but you can pre-download them:

```
   python -m spacy download en_core_web_sm
   python -m spacy download es_core_news_sm
   # Add other languages as needed

   python -m stanza.download en
   python -m stanza.download es
   # Add other languages as needed
```

1. **Run the application** :

```
   streamlit run app.py
```

## Usage

1. Select a language from the dropdown menu.
2. Enter text in the text area or upload a text file.
3. Click "Analyze" to process the text.
4. View the results in the different tabs:
   * Dependency Tree: Shows relationships between words
   * Constituency Tree: Shows hierarchical sentence structure
   * Grammatical Rules: Displays extracted grammatical patterns with explanations

## Extending the Application

### Adding More Languages

To add support for additional languages, update the `LANGUAGES` dictionary in the code with appropriate model codes for spaCy, Stanza, and NLTK.

### Improving Parsing Accuracy

The current implementation uses default models. For production use, consider:

* Using larger, more accurate language models
* Implementing custom parsing logic for specific language features
* Training domain-specific models for specialized text

## Notes

* The constituency parsing feature provides a simplified example visualization. For production use, you would need to implement a more sophisticated parser.
* PDF parsing is mentioned but not fully implemented in this demo.
* Processing very long texts may cause performance issues - consider implementing chunking for larger documents.
