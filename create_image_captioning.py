import base64
import os
import time
from typing import Tuple

import openai
import pandas as pd
from tqdm import tqdm

GPT4_MODEL_NAME = "gpt-4-0125-preview"
MAX_TOKENS_AMOUNT = 128000
EXTRA_SECURITY_GAP = 100
MAX_OUTPUT_TOKENS_AMOUNT = 4096
GPT4_VISION_MODEL_NAME = "gpt-4-vision-preview"


class ImageCaptionsCreator:
    def create_image_captions(self, photo_path) -> Tuple[str, float, int]:
        if not photo_path:
            return "", 0.0, 0

        base64_image = self._encode_image(photo_path)
        messages = [
            {"role": "user",
             "content": [
                 {
                     "type": "text",
                     "text": "Please describe in as many details as possible what is in the photo"
                 },
                 {
                     "type": "image_url",
                     "image_url": {
                         "url": f"data:image/jpeg;base64,{base64_image}"
                     }
                 }
             ]
             }
        ]
        start_time = time.time()
        try:
            response = openai.ChatCompletion.create(
                model=GPT4_VISION_MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=MAX_OUTPUT_TOKENS_AMOUNT - EXTRA_SECURITY_GAP,
                n=1
            )
            generation_time = time.time() - start_time
            content = response.choices[0]['message']['content']
            completion_tokens = response['usage']['completion_tokens']
            return content, generation_time, completion_tokens

        except openai.error.RateLimitError as e:

            print("Rate limit hit, waiting 10s before retrying...")
            time.sleep(10)  # Wait 10 seconds before retrying, adjust based on API feedback

        except Exception as e:
            print(f"An error occurred: {e}")
            return "", 0.0, 0

    @staticmethod
    def _encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def create_photos_paths(photos_path: str):
        photos = []
        for photo in os.listdir(photos_path):
            photo_path = os.path.join(photos_path, photo)
            photos.append(photo_path)
        return photos


if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    photos_path = "resources/photos"
    photos_captions_creator = ImageCaptionsCreator()
    photos = photos_captions_creator.create_photos_paths(photos_path)
    rows = []
    for i, photo in tqdm(enumerate(photos), total=len(photos)):
        photo_name = os.path.basename(photo)
        photo_caption, gen_time, token_count = photos_captions_creator.create_image_captions(photo)
        print(f"Photo {i + 1} caption: {photo_caption}")
        rows.append([photo_name, photo_caption, gen_time, token_count])

    df = pd.DataFrame(rows, columns=["Photo Name", "photo_caption", "generation_time", "completion_tokens"])

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    df.to_csv("outputs/photos_captions.csv", index=False)