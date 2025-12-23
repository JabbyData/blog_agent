# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "happytransformer",
#     "langchain-huggingface",
#     "streamlit",
#     "torch",
#     "transformers",
# ]
# ///
import streamlit as st
import torch
from langchain_huggingface import HuggingFacePipeline
from tools import load_models
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


@st.cache_resource
def load_corrector(query_model_path):
    tokenizer = AutoTokenizer.from_pretrained(query_model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(query_model_path)
    pipe = pipeline(
        task="text2text-generation", model=model, tokenizer=tokenizer, device="cuda"
    )

    return pipe


def summarize_query(corrector: pipeline):
    user_query = st.text_area("Quel contenu souhaitez-vous explorer ?")

    if st.button("Analyse"):
        if user_query.strip() == "":
            st.warning("Merci de rentrer du contenu dans le champs de recherche.")
        else:
            with st.spinner("Analyse de votre demande en cours ..."):
                try:
                    response = corrector(f"grammar: {user_query}")
                    st.subheader("Voici une version corrigée de votre phrase :")
                    st.success(response)
                except Exception as e:
                    st.error(f"Une erreur est survenue : {e}")


def main(paths: dict) -> None:
    st.title("Welcome to AlgoBlog :rocket:", text_alignment="center")
    st.header(body="Your blog feeding partner :writing_hand: ", text_alignment="center")
    corrector = load_corrector(paths["query_corrector_dir"])
    summarize_query(corrector=corrector)


if __name__ == "__main__":
    paths = load_models()
    main(paths)
