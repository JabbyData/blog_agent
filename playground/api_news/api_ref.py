# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "newsapi-python",
#     "requests>=2.32.5",
# ]
# ///
# import requests
import json
import os

from newsapi import NewsApiClient, const


def main(config_path: str) -> None:
    # Init
    with open(config_path) as f:
        configs = json.load(f)

    newsapi = NewsApiClient(api_key=configs["newsapi_api_key"])

    # sources
    # print(const.countries)
    ## Test news with French sources
    sources = newsapi.get_sources(country="fr")["sources"]
    restricted_sources = ",".join([source["id"] for source in sources])

    q = "tech"
    from_param = "2025-11-15"
    to = "2025-12-15"
    language = "en"
    sort_by = "relevancy"

    # /v2/everything
    all_articles = newsapi.get_everything(
        q=q,
        sources=restricted_sources,
        from_param=from_param,
        to=to,
        language=language,
        sort_by=sort_by,
    )

    print(all_articles["articles"][0])


if __name__ == "__main__":
    config_path = os.path.join(os.getcwd(), "configs.json")
    main(config_path)
