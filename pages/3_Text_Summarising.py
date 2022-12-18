import torch
from transformers import pipeline
import streamlit as st
import pathlib


st.header("Summarising text based on Transformer-based pretrained model")
value = st.text_area("Enter the text you want to summarise here.")
button= st.button("Summarise text")
if button:
    summarizer = pipeline(
    "summarization",
    "pszemraj/long-t5-tglobal-base-16384-book-summary"
    )
    result = summarizer(value)
    st.write(result[0]["summary_text"])
