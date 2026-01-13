# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mistral-inference",
#     "torch",
#     "transformers",
# ]
# ///
import os

from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_inference.generate import generate
from mistral_inference.transformer import Transformer


def main(analyzer_dir) -> None:
    tokenizer_dir_path = os.path.join(analyzer_dir, "tokenizer.model.v3")
    tokenizer = MistralTokenizer.from_file(tokenizer_dir_path)
    model = Transformer.from_folder(analyzer_dir)

    # completion_request = ChatCompletionRequest(
    #     messages=[
    #         UserMessage(
    #             content=(
    #                 "TASK: Rewrite the following user query to correct all grammatical and spelling errors, "
    #                 "then provide keywords summarizing the query.\n\n"
    #                 "FORMAT:\n"
    #                 "OUTPUT: [Corrected and concise summary]\n\n"
    #                 "CONSTRAINTS:\n"
    #                 "- No conversational filler.\n"
    #                 "- The summary must be under 20 words.\n"
    #                 "- DO NOT USE VERBS.\n"
    #                 "- Output should start with OUTPUT.\n"
    #                 "- Output should end with a point.\n"
    #                 "- Make it simple and neutral.\n\n"
    #                 "QUERY: " + "Find articles about tech stratups in Europe."
    #             )
    #         ),
    #     ]
    # )

    user_draft = """## La French Tech: The New Golden Age of French Innovation
    A decade ago, Fronce was primarily seen as a land of traditional industries, often perceived ase being slowed by some redundancy in the area of the burocracy. Today, the landscape has radically shifted. From Paris to Lyon, and Montpellier to Lille, the startup ecosystem—united under the 'La French Tech' label—has become one of the most dynamic in Europe.
    ### A Supported and Structured Ecosystem
    The successs of French startups is not what one would think of as an accident. It his built on strong polithical will and unique public support, notably through Bpifrance. This public investment bank acts as a catalyst by financing innovation and de-risking investments for private players.
    Furthermore, France has invested in world-class infrastructure. Station F, located in Paris, is currently the world’s largest startup campus, housing over 1,000 startups under one roof. This venue symbolizes the transformation of the capital into a global tech hub.
    ### From 'Unicorns' to 'Centaurs'
    France hit its government-mandated target two years early: the first goal is about surpassing 25 unicorns (an unicorn is considered as an unquoted companie valued at over $1 billion). Names like Back Market, Doctolib, ManoMano, and ContentSquare have become leaders in their respective sectors.
    However, beyond mere valuation, the focus is about shifting toward 2 main horizons that are profitability and impact. We are seeing the rise of GreenTech and decarbonized industrial startups, directly addressing current climate challenges.
    ### New Frontiers: AI and Sovereignty
    Artificial Intelligence (AI) is the new batlefield. With champions like Mistral AI, France is prouving it can compete with American and Chinese giants. The stakes are twofold: attracting international talent and ensuring European technological sovereignty.
    Despite this euphoria, challenges remain:
    * Late-stage Funding: While early seed rounds are plentiful, massive funding rounds of several hundred million euros still rely heavily on foreign capital.
    * Diversity and Parity: Women remain underrepresented among startup founders, representing a significant untapped growth lever.
    ### Conclusion
    The 'Startup Nation' is no longer just a slogan; it is an economic reality creating thousands of jobs every year. France has successfully fostered an environment where failure is less stigmatized and global ambition is the norm. The next challenge will be scale: transforming these tech successes into enduring industrial giants."""

    prompt = (
        "TASK: Rewrite the following user draft to correct all grammatical and spelling errors, "
        "then summarize the corrected draft.\n\n"
        "FORMAT:\n"
        "OUTPUT: [Corrected summary]\n\n"
        "CONSTRAINTS:\n"
        "- No conversational filler.\n"
        "- Output should start with OUTPUT.\n"
        "QUERY: " + user_draft
    )
    completion_request = ChatCompletionRequest(
        messages=[
            UserMessage(content=prompt),
        ]
    )

    tokens = tokenizer.encode_chat_completion(completion_request).tokens

    out_tokens, _ = generate(
        [tokens],
        model,
        max_tokens=600,
        temperature=0.0,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
    )
    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

    print(result)


if __name__ == "__main__":
    analyzer_dir = os.path.join(os.getcwd(), "agents", "query_analyzer")
    main(analyzer_dir)
