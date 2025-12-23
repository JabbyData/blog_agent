# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mistral-inference",
#     "torch",
#     "transformers",
# ]
# ///
import os

import torch
from mistral_common.protocol.instruct.messages import SystemMessage, UserMessage
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
                content="You are a pro writer, correct mistakes and sum up the following : I want to write tech startups, find me information about new companies in Europe over the last decade."
            ),
        ]
    )

    # TODO : learn to instruct Mistral Instruct

    tokens = tokenizer.encode_chat_completion(completion_request).tokens

    out_tokens, _ = generate(
        [tokens],
        model,
        max_tokens=64,
        temperature=0.0,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
    )
    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

    print(result)


if __name__ == "__main__":
    analyzer_dir = os.path.join(os.getcwd(), "agents", "query_analyzer")
    main(analyzer_dir)
