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

# Disable file watching in Streamlit to avoid the PyTorch conflict
os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"

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

# Function to create a pure HTML constituency tree 
def get_constituency_parse_html(text, lang):
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
        margin-bottom: 15px;
        text-align: center;
    }
    .connector {
        position: relative;
        height: 15px;
    }
    .line {
        position: absolute;
        width: 1px;
        background-color: #ccc;
        left: 50%;
        top: 0;
        height: 15px;
    }
    </style>
    <div class="tree">
        <div class="level">
            <div class="node">S</div>
        </div>
        <div class="connector"><div class="line"></div></div>
        <div class="level">
    """
    
    # Second level: Basic phrase structure
    middle = len(words) // 2
    if middle == 0:
        html += '<div class="node">NP</div>'
    else:
        html += '<div class="node">NP</div><div class="node">VP</div>'
    
    html += """
        </div>
        <div class="connector"><div class="line"></div></div>
        <div class="level">
    """
    
    # Third level: Part of speech
    for i, word in enumerate(words):
        if i < middle:
            category = "DT" if i == 0 else "NN"
        else:
            category = "V" if i == middle else "NN"
        html += f'<div class="node">{category}</div>'
    
    html += """
        </div>
        <div class="connector"><div class="line"></div></div>
        <div class="level">
    """
    
    # Fourth level: Words
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
    uploaded_file = st.file_uploader("Or upload a text document", type=["txt", "md"])
    
    if uploaded_file is not None:
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
                st.subheader("Constituency Parse")
                html_tree = get_constituency_parse_html(text_input, language)
                st.markdown(html_tree, unsafe_allow_html=True)
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