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

    completion_request = ChatCompletionRequest(
        messages=[
            UserMessage(
                content=(
                    "TASK: Rewrite the following user query to correct all grammatical and spelling errors, "
                    "then provide a one-sentence summary of the core intent.\n\n"
                    "FORMAT:\n"
                    "REVISED: [Corrected text]\n"
                    "SUMMARY: [Concise summary]\n\n"
                    "CONSTRAINTS:\n"
                    "- No conversational filler.\n"
                    "- The summary must be under 20 words.\n"
                    "- Maintain the original tone.\n\n"
                    "QUERY: "
                    "I want ot right about news in the worlde. Find me information about tech startups in Urope over the last months. I want articles in French."
                )
            ),
        ]
    )

    tokens = tokenizer.encode_chat_completion(completion_request).tokens

    out_tokens, _ = generate(
        [tokens],
        model,
        max_tokens=100,
        temperature=0.0,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
    )
    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

    print(result)


if __name__ == "__main__":
    analyzer_dir = os.path.join(os.getcwd(), "agents", "query_analyzer")
    main(analyzer_dir)
