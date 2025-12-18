# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "streamlit",
# ]
# ///
import streamlit as st


def get_user_query() -> str:
    user_query = st.text_area(label="What would you like to explore ?")

    if not user_query:
        st.warning("Please provide information about your query above.")
    else:
        st.success("Perfect, let me process your query ...")
        return user_query

    return None


def main() -> None:
    st.title("Welcome to AlgoBlog :rocket:", text_alignment="center")
    st.header(body="Your blog feeding partner :writing_hand: ", text_alignment="center")
    user_query = get_user_query()

    if user_query:
        print("User query loaded !")


if __name__ == "__main__":
    main()
