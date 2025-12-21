# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain-huggingface",
#     "streamlit",
#     "torch",
#     "transformers",
# ]
# ///
import streamlit as st
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline


@st.cache_resource  # reuse pipe between refreshes
def load_summarizer():
    pipe = pipeline("summarization", model="Falconsai/text_summarization", device=0)
    return HuggingFacePipeline(pipeline=pipe)


def summarize_query(query_llm: HuggingFacePipeline):
    user_query = st.text_area("Quel contenu souhaitez-vous explorer ?")

    if st.button("Analyse"):
        if user_query.strip() == "":
            st.warning("Merci de rentrer du contenu dans le champs de recherche.")
        else:
            with st.spinner("Analyse de votre demande en cours ..."):
                try:
                    response = query_llm.invoke(user_query)
                    st.subheader("Voici une vue synthétique de votre recherche :")
                    st.success(response)
                except Exception as e:
                    st.error(f"Une erreur est survenue : {e}")


def main() -> None:
    st.title("Welcome to AlgoBlog :rocket:", text_alignment="center")
    st.header(body="Your blog feeding partner :writing_hand: ", text_alignment="center")
    query_llm = load_summarizer()
    summarize_query(query_llm=query_llm)


if __name__ == "__main__":
    main()
