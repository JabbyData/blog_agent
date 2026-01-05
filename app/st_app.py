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
from datetime import date, timedelta

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
    st.subheader("1. Analyse de votre requête :speech_balloon:")
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
                    st.session_state["processed_query"] = response
                except Exception as e:
                    st.error(f"Une erreur est survenue : {e}")


def extract_search_criteria():
    query = st.session_state.get("processed_query")

    if query:
        st.divider()
        st.subheader("2. Paramètres de recherche :dart:")
        st.info(f"Requête analysée : **{query}**")

        with st.form("criteria_form"):
            st.write("Affinez votre recherche. Laissez vide si pas de préférence.")
            sources = st.text_input(label="Sources", placeholder="Ex. bbc-news, l'écho")

            col1, col2 = st.columns(2)
            dt_start = col1.date_input(
                label="Date de début",
                min_value=date.today() - timedelta(days=30),
            )
            dt_end = col2.date_input(
                label="Date de fin",
                max_value=date.today(),
            )
            language = st.text_input(label="Language", placeholder="Ex. fr")

            submit = st.form_submit_button("Confirmer")

            if submit:
                search_criteria = {
                    "q": query,
                    "sources": sources,
                    "from_param": dt_start.__str__(),
                    "to": dt_end.__str__(),
                    "language": language,
                }
                return search_criteria
    return None


def main() -> None:
    # Intro
    st.title("Welcome to AlgoBlog :rocket:", text_alignment="center")
    st.header(body="Your blog feeding partner :writing_hand: ", text_alignment="center")
    configs_path = os.path.join(os.getcwd(), "configs.json")

    # Query analyzer
    analyzer = load_analyzer(configs_path)
    treat_user_query(analyzer=analyzer)

    # URL retrieval
    search_criteria = extract_search_criteria()

    if search_criteria:
        print(search_criteria)

    # TODO : search with given information


if __name__ == "__main__":
    main()
