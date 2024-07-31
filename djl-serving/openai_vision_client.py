import base64

import requests
from openai import OpenAI
from transformers import AutoTokenizer

from engines.python.setup.djl_python.chat_completions.chat_utils import parse_chat_completions_request

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
image_url = "https://resources.djl.ai/images/dog_bike_car.jpg"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)


# Use base64 encoded image in the payload
def encode_image_base64_from_url(image_url: str) -> str:
    """Encode an image retrieved from a remote url to base64 format."""
    with requests.get(image_url) as response:
        response.raise_for_status()
        base64_image = base64.b64encode(response.content).decode('utf-8')
    return base64_image


image_base64 = encode_image_base64_from_url(image_url=image_url)
sample_messages = [{
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "What’s in this image?"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            },
        },
    ],
}]
sample_input_map = {'messages': sample_messages, 'model': ""}

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-v1.6-34b-hf", use_fast=False)
    inputs, params = parse_chat_completions_request(sample_input_map,
                                                    is_rolling_batch=True,
                                                    tokenizer=tokenizer)
    print(inputs)
    images = params.pop("images", None)
    for image in images:
        print(image)
    print(params)
r