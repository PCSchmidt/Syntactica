import os
import streamlit as st
import spacy
from spacy import displacy
import nltk
from nltk import Tree
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd

# Disable the PyTorch warning by setting environment variable
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# Download necessary NLTK data
nltk.download('punkt')

# Initialize language resources
@st.cache_resource
def load_spacy_model(lang_code):
    try:
        return spacy.load(lang_code)
    except:
        spacy.cli.download(lang_code)
        return spacy.load(lang_code)

# Supported languages and their codes
LANGUAGES = {
    "English": {"spacy": "en_core_web_sm", "nltk": "english"},
    "Spanish": {"spacy": "es_core_news_sm", "nltk": "spanish"},
    "French": {"spacy": "fr_core_news_sm", "nltk": "french"},
    "German": {"spacy": "de_core_news_sm", "nltk": "german"},
    "Italian": {"spacy": "it_core_news_sm", "nltk": "italian"},
    "Portuguese": {"spacy": "pt_core_news_sm", "nltk": "portuguese"},
    "Dutch": {"spacy": "nl_core_news_sm", "nltk": "dutch"},
    "Greek": {"spacy": "el_core_news_sm", "nltk": None},
    "Russian": {"spacy": "ru_core_news_sm", "nltk": "russian"},
    "Chinese": {"spacy": "zh_core_web_sm", "nltk": "chinese"},
    "Japanese": {"spacy": "ja_core_news_sm", "nltk": "japanese"},
    "Arabic": {"spacy": "ar_core_news_sm", "nltk": "arabic"}
}

# Function to get dependency parse with spaCy
def get_dependency_parse_spacy(text, lang):
    nlp = load_spacy_model(LANGUAGES[lang]["spacy"])
    doc = nlp(text)
    svg = displacy.render(doc, style="dep", options={"compact": True})
    return svg

# Function to create a sentence-specific constituency tree based on input text
def create_sample_tree_for_text(text):
    # This is a simplified approach that creates a basic tree
    # For a real parser, you would use a trained model
    words = text.split()
    if len(words) <= 1:
        return Tree('S', [Tree('NP', [Tree('NN', [text])])])
    
    # Try to identify a simple subject-verb-object structure
    if len(words) <= 3:
        return Tree('S', [
            Tree('NP', [Tree('NN', [words[0]])]),
            Tree('VP', [Tree('V', [words[1]])] + 
                 ([Tree('NP', [Tree('NN', [words[2]])])] if len(words) > 2 else []))
        ])
    
    # More complex sentence
    middle = len(words) // 2
    return Tree('S', [
        Tree('NP', [Tree('NN', [' '.join(words[:middle//2])])]),
        Tree('VP', [
            Tree('V', [words[middle]]),
            Tree('PP', [
                Tree('IN', ['with']),
                Tree('NP', [Tree('NN', [' '.join(words[middle+1:])])])
            ])
        ])
    ])

# Function to get constituency parse with NLTK
def get_constituency_parse_nltk(text, lang):
    if LANGUAGES[lang]["nltk"]:
        try:
            # Create a custom tree based on the input text
            tree = create_sample_tree_for_text(text)
            
            # Draw the tree
            plt.figure(figsize=(12, 6))
            tree.draw()
            
            # Save the figure to a buffer
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            
            # Convert to base64 for display
            img_str = base64.b64encode(buf.read()).decode()
            return f'<img src="data:image/png;base64,{img_str}" />'
        except Exception as e:
            return f"<p>Error generating constituency parse: {str(e)}</p><p>Using fallback visualization...</p>" + create_fallback_tree_visualization(text)
    else:
        return "Constituency parsing not available for this language with NLTK."

# Create a simple HTML-based fallback tree visualization
def create_fallback_tree_visualization(text):
    words = text.split()
    if not words:
        return "<p>No text to parse</p>"
    
    html = """
    <style>
    .tree {
        font-family: Arial, sans-serif;
    }
    .node {
        padding: 5px 10px;
        border: 1px solid #ccc;
        display: inline-block;
        margin: 2px;
        background-color: #f8f9fa;
    }
    .level {
        margin-bottom: 10px;
        text-align: center;
    }
    </style>
    <div class="tree">
        <div class="level">
            <div class="node">S</div>
        </div>
        <div class="level">
            <div class="node">NP</div>
            <div class="node">VP</div>
        </div>
        <div class="level">
    """
    
    middle = len(words) // 2
    for i, word in enumerate(words):
        if i < middle:
            category = "DT" if i == 0 else "NN"
        else:
            category = "V" if i == middle else "NN"
        html += f'<div class="node">{category}</div>'
    
    html += """
        </div>
        <div class="level">
    """
    
    for word in words:
        html += f'<div class="node">{word}</div>'
    
    html += """
        </div>
    </div>
    """
    return html

# Function to get grammatical rules with spaCy
def get_grammatical_rules(text, lang):
    nlp = load_spacy_model(LANGUAGES[lang]["spacy"])
    doc = nlp(text)
    
    rules = []
    
    # Extract dependency relationships and POS patterns
    for token in doc:
        rule = {
            "Word": token.text,
            "POS": token.pos_,
            "Description": get_pos_description(token.pos_),
            "Dependency": token.dep_,
            "Head": token.head.text if token.dep_ != "ROOT" else "None"
        }
        rules.append(rule)
    
    # If no rules were extracted, add a dummy rule to show something
    if not rules:
        rules.append({
            "Word": text,
            "POS": "Unknown",
            "Description": "Could not parse",
            "Dependency": "Unknown",
            "Head": "None"
        })
    
    return pd.DataFrame(rules)

# Function to get descriptions for POS tags
def get_pos_description(pos_tag):
    pos_descriptions = {
        "ADJ": "Adjective",
        "ADP": "Adposition (preposition or postposition)",
        "ADV": "Adverb",
        "AUX": "Auxiliary verb",
        "CCONJ": "Coordinating conjunction",
        "DET": "Determiner",
        "INTJ": "Interjection",
        "NOUN": "Noun",
        "NUM": "Numeral",
        "PART": "Particle",
        "PRON": "Pronoun",
        "PROPN": "Proper noun",
        "PUNCT": "Punctuation",
        "SCONJ": "Subordinating conjunction",
        "SYM": "Symbol",
        "VERB": "Verb",
        "X": "Other"
    }
    return pos_descriptions.get(pos_tag, "Unknown part of speech")

# Main Streamlit app
def main():
    st.title("Syntactica - Multilingual Syntax Analyzer")
    st.write("Analyze text in multiple languages to generate syntax trees and understand grammatical rules.")
    
    # Language selection
    language = st.selectbox("Select Language", list(LANGUAGES.keys()))
    
    # Text input
    default_text = "They not only signed the letter."
    text_input = st.text_area("Enter text to analyze:", value=default_text, height=150)
    
    # File upload option
    uploaded_file = st.file_uploader("Or upload a text document", type=["txt", "md", "pdf"])
    
    if uploaded_file is not None:
        # Process the uploaded file
        if uploaded_file.type == "application/pdf":
            # For a real app, you would use a PDF parser like PyPDF2 or pdfminer.six
            st.warning("PDF parsing is not implemented in this demo. Please paste text directly.")
        else:
            text_input = uploaded_file.read().decode("utf-8")
    
    if st.button("Analyze") or True:  # Auto-analyze on load
        if text_input:
            st.subheader("Analysis Results")
            
            # Create tabs for different analyses
            tabs = st.tabs(["Dependency Tree", "Constituency Tree", "Grammatical Rules"])
            
            with tabs[0]:
                st.subheader("Dependency Parse (spaCy)")
                try:
                    svg = get_dependency_parse_spacy(text_input, language)
                    st.markdown(svg, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating dependency parse: {str(e)}")
                    st.info("Try with a shorter sentence or a different language.")
            
            with tabs[1]:
                st.subheader("Constituency Parse (NLTK)")
                tree_img = get_constituency_parse_nltk(text_input, language)
                st.markdown(tree_img, unsafe_allow_html=True)
                st.caption("Note: This is a simplified tree visualization. A production app would use a more sophisticated parsing model.")
            
            with tabs[2]:
                st.subheader("Grammatical Rules")
                rules_df = get_grammatical_rules(text_input, language)
                st.dataframe(rules_df)
                
                # Explanation of common dependencies
                st.subheader("Common Dependency Relations")
                dependencies = {
                    "nsubj": "Nominal subject - the subject of the sentence",
                    "obj": "Object - the direct object of the verb",
                    "det": "Determiner - articles and other determiners",
                    "amod": "Adjectival modifier - adjective modifying a noun",
                    "advmod": "Adverbial modifier - adverb modifying a verb",
                    "aux": "Auxiliary - helping verb",
                    "conj": "Conjunct - connected by a coordinating conjunction",
                    "prep": "Preposition - introduces a prepositional phrase",
                    "compound": "Compound - words that function together as a unit",
                    "ROOT": "Root - the central word of the sentence or clause"
                }
                for dep, desc in dependencies.items():
                    st.write(f"**{dep}**: {desc}")

if __name__ == "__main__":
    main()