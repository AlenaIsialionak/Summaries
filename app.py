import streamlit as st
import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, GPT2Tokenizer, T5ForConditionalGeneration
import torch
import os
from typing import Tuple, Union, Optional
from uploader import read_file



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ["TRANSFORMERS_CACHE"] = "models"
os.makedirs("models", exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_model_and_tokenizer(
    model_name: str = "facebook/bart-large-cnn",
    model_choice: str = 'English'
) -> Union[Tuple[AutoModelForSeq2SeqLM, AutoTokenizer], Tuple[T5ForConditionalGeneration, GPT2Tokenizer], Tuple[None, None]]:
    
    """
    Loads a pre-trained model and tokenizer.

    :param model_name: str - Name of English model (default: "facebook/bart-large-cnn").
    :param model_choice: str - 'English' or 'Russian' (default: 'English').

    :return: tuple(model, tokenizer) or tuple(None, None) - Model and tokenizer, or None on error."""

    try:
        if model_choice == 'English':
            logging.info(f"Loading model: {model_name}")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logging.info(f"Model {model_name} loaded successfully.")
            return model, tokenizer
        
        if model_choice == 'Russian':
            logging.info(f"Loading Russian model")
            try:
                tokenizer = GPT2Tokenizer.from_pretrained('RussianNLP/FRED-T5-Summarizer', eos_token='</s>')
                model = T5ForConditionalGeneration.from_pretrained('RussianNLP/FRED-T5-Summarizer')
                model.to(device)
                return model, tokenizer
            except Exception as e:
                logging.error(f"Russian model loading failed: {str(e)}")
                return None, None
        
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        st.error(f"Model loading failed: {str(e)}")
        return None, None

def sliding_window_summarize(
    text: str,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    model_choice: str,
    window_size: int = 512,
    overlap: int = 128,
    max_length: int = 150
) -> str:
    """
    Summarizes a large text using a sliding window approach.

    :param text: str - The input text to summarize.
    :param model: AutoModelForSeq2SeqLM - The pre-trained summarization model.
    :param tokenizer: AutoTokenizer - The tokenizer corresponding to the model.
    :param model_choice: str -  'English' or 'Russian', dictates summarization method.
    :param window_size: int - The size of the sliding window (default: 512 words).
    :param overlap: int - The number of overlapping words between adjacent windows (default: 128).
    :param max_length: int - The maximum length of the generated summary for each window (default: 150 tokens).

    :return: str - The concatenated summaries of each window, forming the overall summary of the input text.
    """
    
    words = text.split()
    step = window_size - overlap
    chunks = [' '.join(words[i:i + window_size]) for i in range(0, len(words), step)]
    
    chunk_summaries = []
    for chunk in chunks:
        try:
            if model_choice == 'English':
                summarizer = pipeline(
                    "summarization",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )
                summary = summarizer(
                    chunk,
                    max_length=max_length,  
                    min_length=max(30, max_length // 3),
                    do_sample=False
                )[0]['summary_text']
            else:
                inputs = tokenizer(
                    chunk, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(device)
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_length,
                    attention_mask=inputs.attention_mask,
                    eos_token_id=tokenizer.eos_token_id,
                    num_beams=5,
                    min_new_tokens=30,
                    do_sample=True,
                    no_repeat_ngram_size=4,
                    top_p=0.9
                )
                summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            chunk_summaries.append(summary)
        
        except Exception as e:
            logging.error(f"Error summarizing chunk: {str(e)}")
            continue
    
    return ' '.join(chunk_summaries)

def summarize_text(
    text: str,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    model_choice: str,
    max_length: int = 150
) -> str:
    """
    Summarizes text based on its length, employing different summarization methods.

    :param text: str - The input text to be summarized.
    :param model: AutoModelForSeq2SeqLM - The pre-trained summarization model.
    :param tokenizer: AutoTokenizer - The tokenizer corresponding to the model.
    :param model_choice: str - 'English' or 'Russian', dictating the summarization approach.
    :param max_length: int - The maximum length of the generated summary (default: 150 tokens).

    :return: str - The generated summary. Returns an empty string if the input text is empty,
                    a summary generated by sliding window or direct summarization based on
                    the input text's length, or an error message if summarization fails.
                    Also returns a message if text is too short to summarize.
    """

    if not text.strip():
        return ""
    
    word_count = len(text.split())
    
    if word_count > 1000:
        return sliding_window_summarize(
            text, 
            model, 
            tokenizer, 
            model_choice, 
            max_length=max_length)
    
    elif 30 < word_count <= 1000:
        try:
            if model_choice == 'English':
                summarizer = pipeline(
                    "summarization",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )
                return summarizer(
                    text,
                    max_length=max_length,
                    min_length=max(30, max_length // 3),
                    do_sample=False
                )[0]['summary_text']
            else: 
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).to(device)
                
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    eos_token_id=tokenizer.eos_token_id,
                    num_beams=5,
                    min_new_tokens=30,
                    max_new_tokens=max_length,
                    do_sample=True,
                    no_repeat_ngram_size=4,
                    top_p=0.9
                )
                return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        except Exception as e:
            logging.error(f"Summarization error: {str(e)}")
            return f"Error during summarization: {str(e)}"
    
    else:
        return "Text is too short for summarization (minimum 30 words required)"


MODELS = {
    "English": "facebook/bart-large-cnn",
    "Russian": "RussianNLP/FRED-T5-Summarizer"
}

model_choice = st.selectbox("Select model:", list(MODELS.keys()))
model_name = MODELS[model_choice]

model, tokenizer = load_model_and_tokenizer(model_name, model_choice)
if model is None or tokenizer is None:
    st.error("Failed to load models. Please check your internet connection and try again.")
    st.stop()  


uploaded_file = st.file_uploader("Upload file", type=["txt", "pdf", "docx"])

text = ""
if uploaded_file is not None:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    text = read_file(uploaded_file.name)
    if text:
        st.success("File uploaded and read successfully!")
    else:
        st.error("Failed to read file.")
    os.remove(uploaded_file.name)

max_length = st.slider("Max summary length:", 50, 500, 150)

if st.button("Generate Summary") and text.strip():
    if model and tokenizer:
        with st.spinner("Generating summary..."):
            result = summarize_text(text, model, tokenizer, model_choice, max_length)
            if result:
                st.subheader("Result:")
                st.info(result)
    else:
        st.error("Model not loaded. Please restart the application.")