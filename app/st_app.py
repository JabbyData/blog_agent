# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "happytransformer",
#     "langchain-huggingface",
#     "newsapi-python",
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
from newsapi.newsapi_client import NewsApiClient

from agents.query_analyzer.query_analyzer import QueryAnalyzer


@st.cache_resource
def load_analyzer(configs_path):
    with open(configs_path, "r") as f:
        configs = json.load(f)
    analyzer = QueryAnalyzer(
        tokenizer_path=configs["query_analyzer"]["tokenizer_path"],
        model_path=configs["query_analyzer"]["model_path"],
    )
    st.session_state["analyzer"] = analyzer


def treat_user_query():
    if st.session_state["analyzer"]:
        analyzer = st.session_state["analyzer"]

        st.subheader("1. Analyse de votre requête :speech_balloon:")
        user_query = st.text_area("Quel contenu souhaitez-vous explorer ?")

        if st.button("Analyser"):
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
        st.info(f"Requête analysée : {query}")

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
                    "q": query if query != "" else None,
                    "sources": sources if sources != "" else None,
                    "from_param": dt_start.__str__(),
                    "to": dt_end.__str__(),
                    "language": language if language != "" else None,
                }
                st.session_state["search_criteria"] = search_criteria


def get_all_articles():
    search_criteria = st.session_state.get("search_criteria")

    if search_criteria:
        with st.spinner("Chargement en cours ..."):
            config_path = os.path.join(os.getcwd(), "configs.json")
            try:
                with open(config_path) as f:
                    configs = json.load(f)
                    news_api_key = configs["newsapi"]["api_key"]
            except Exception as e:
                print("An error occured : ", e)

            newsapi = NewsApiClient(news_api_key)
            all_articles = newsapi.get_everything(
                q=search_criteria["q"],
                sources=search_criteria["sources"],
                from_param=search_criteria["from_param"],
                to=search_criteria["to"],
                language=search_criteria["language"],
                sort_by="relevancy",
            )

            st.session_state["all_articles"] = all_articles["articles"]


def display_articles():
    all_articles = st.session_state.get("all_articles")

    if all_articles:
        st.divider()
        st.subheader("3. Collecte des références :basket:")
        BATCH_SIZE = 10  # nb max elt per page
        if "display_limit" not in st.session_state:
            st.session_state["display_limit"] = BATCH_SIZE

        articles_container = st.container()

        with articles_container:
            for i in range(
                st.session_state["display_limit"] - BATCH_SIZE,
                st.session_state["display_limit"],
            ):
                if i < len(all_articles):
                    article = all_articles[i]

                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"##### {article['title']}")
                        desc = article["description"]
                        if len(desc) > 200:
                            desc = desc[:200] + "..."
                        st.write(desc)
                    with col2:
                        st.write("")
                        st.link_button(
                            "Ouvrir :link:", article["url"], use_container_width=True
                        )
        col3, col4 = st.columns(2)
        with col3:
            if st.button(
                ":arrow_left: Afficher les articles précédents ",
                use_container_width=True,
            ):
                st.session_state["display_limit"] = max(
                    BATCH_SIZE, st.session_state["display_limit"] - BATCH_SIZE
                )
                st.rerun()
        with col4:
            if st.button(
                "Afficher les articles suivants :arrow_right:", use_container_width=True
            ):
                st.session_state["display_limit"] = min(
                    len(all_articles), st.session_state["display_limit"] + BATCH_SIZE
                )
                st.rerun()


def treat_user_draft():
    if (
        "display_limit" in st.session_state
    ):  # activation only once first articles appeared
        st.divider()
        st.subheader("4. Analyse de votre article :thought_balloon:")
        user_draft = st.text_area("Merci de rentrer votre contenu ici.")

        if st.button("Valider"):
            if user_draft.strip() == "":
                st.warning("Merci de rentrer des du contenu avant de l'analyser.")
            else:
                analyzer = st.session_state["analyzer"]
                with st.spinner("Analyse de votre demande en cours ..."):
                    try:
                        response = (
                            analyzer.process_user_draft(user_draft)
                            .split("OUTPUT:")[1]
                            .split(".")[0]
                        )
                        st.session_state["processed_draft"] = response
                        print(response)
                    except Exception as e:
                        st.error(f"Une erreur est survenue : {e}")

        if "processed_draft" in st.session_state:
            st.info(
                f"Article synthétisé : \n {st.session_state.get('processed_draft')}"
            )


def main() -> None:
    # Intro
    st.title("Welcome to AlgoBlog :rocket:", text_alignment="center")
    st.header(body="Your blog feeding partner :writing_hand: ", text_alignment="center")
    configs_path = os.path.join(os.getcwd(), "configs.json")

    # Query analyzer
    load_analyzer(configs_path)
    treat_user_query()

    # URL retrieval
    extract_search_criteria()
    get_all_articles()
    display_articles()

    # Draft enhancement
    treat_user_draft()


if __name__ == "__main__":
    main()
