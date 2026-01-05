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
import json
import os
import sys

# adding root dir to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from agents.query_analyzer.query_analyzer import QueryAnalyzer


@st.cache_resource
def load_analyzer(configs_path):
    with open(configs_path, "r") as f:
        configs = json.load(f)
    analyzer = QueryAnalyzer(
        tokenizer_path=configs["query_analyzer"]["tokenizer_path"],
        model_path=configs["query_analyzer"]["model_path"],
    )
    return analyzer


def treat_user_query(analyzer):
    user_query = st.text_area("Quel contenu souhaitez-vous explorer ?")

    if st.button("Analyse"):
        if user_query.strip() == "":
            st.warning("Merci de rentrer du contenu dans le champs de recherche.")
        else:
            with st.spinner("Analyse de votre demande en cours ..."):
                try:
                    response = (
                        analyzer.process_user_query(user_query)
                        .split("OUTPUT:")[1]
                        .split(".")[0]
                    )
                    st.subheader("Version synthétique de votre recherche :")
                    st.success(response)
                except Exception as e:
                    st.error(f"Une erreur est survenue : {e}")


def main() -> None:
    st.title("Welcome to AlgoBlog :rocket:", text_alignment="center")
    st.header(body="Your blog feeding partner :writing_hand: ", text_alignment="center")
    configs_path = os.path.join(os.getcwd(), "configs.json")
    analyzer = load_analyzer(configs_path)
    treat_user_query(analyzer=analyzer)


if __name__ == "__main__":
    main()
