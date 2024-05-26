import os
from typing import List

import openai
import pandas as pd
from openai.embeddings_utils import get_embedding, distances_from_embeddings
import tiktoken

from Generative_project.chatbot_project.create_image_embeddings import TEXT_EMBEDDING_LARGE_MODEL

SYSTEM_PROMPT = "Answer the user question as detailed as possible. In the end of your answer" \
                "provide a list of the image names you used to answer. If the image isn't " \
                "relevant for the question- ignore it" \
                "Don't include the photos names in your the answers' body, only at the end of it."

PROMPT_TEMPLATE = """
            Answer the question based on the context below, and if the question
            can't be answered based on the context, answer with an empty string.
            Provide your answer in a JSON format like so:
            
            'text' : 'The answer to the question', 
            'image_evidences': ['image1.jpg', 'image2.jpg']
                
            Context: 
        
            {}
        
            ---
        
            Question: {}
            Answer:
        """

COMPLETION_MODEL_NAME = "gpt-4-0125-preview"
MAX_TOKENS_AMOUNT = 128000
EXTRA_SECURITY_GAP = 100
MAX_OUTPUT_TOKENS_AMOUNT = 4096


class ImageQuery:
    def __init__(self, image_embeddings_path: str):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.image_embeddings = pd.read_csv(image_embeddings_path)

    def answer_question(self,
                        question: str,
                        top_k_context_to_use=10,
                        simple_format: bool = False
                        ) -> str:
        """
        Given a question, a dataframe containing rows of text, and a maximum
        number of desired tokens in the messages and response, return the
        answer to the question according to an OpenAI Completion model

        If the model produces an error, return an empty string
        """

        messages = self.create_prompt(question, max_token_count=MAX_TOKENS_AMOUNT,
                                      top_k_context_to_use=top_k_context_to_use,
                                      simple_format=simple_format)

        try:
            response = openai.ChatCompletion.create(
                model=COMPLETION_MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=MAX_OUTPUT_TOKENS_AMOUNT - EXTRA_SECURITY_GAP,
            )
            return response["choices"][0]["message"]['content'].strip()

        except Exception as e:
            print(e)
            return ""

    def create_prompt(self, question, max_token_count, top_k_context_to_use: int, simple_format: bool) -> List[dict]:
        """
        Given a question and a dataframe containing rows of text and their
        embeddings, return a text prompt to send to a Completion model
        """
        tokenizer = tiktoken.get_encoding("cl100k_base")
        prompt_template = PROMPT_TEMPLATE
        current_token_count = len(tokenizer.encode(prompt_template)) + \
                              len(tokenizer.encode(question))

        context = []
        for text, photo_name in \
                self.get_embedding_rows_sorted_by_relevance(question, top_k_context_to_use, simple_format)[
                    ["photo_caption", "Photo Name"]].values:
            text_token_count = len(tokenizer.encode(text))
            current_token_count += text_token_count
            if current_token_count <= max_token_count - EXTRA_SECURITY_GAP:
                context.append(f"{text}, photo_name- {photo_name}")

            else:
                break

        prompt = prompt_template.format("\n\n###\n\n".join(context), question)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return messages

    def get_embedding_rows_sorted_by_relevance(self, query_text: str, top_k_context_to_use: int,
                                               simple_format: bool) -> pd.DataFrame:
        df_copy = self.image_embeddings.copy()
        if simple_format:
            return df_copy

        question_embeddings = get_embedding(query_text, engine=TEXT_EMBEDDING_LARGE_MODEL)
        df_copy["embeddings"] = df_copy["embeddings"].apply(lambda x: eval(x))
        df_copy["distances"] = distances_from_embeddings(
            question_embeddings,
            df_copy["embeddings"].values,
            distance_metric="cosine"
        )
        return df_copy.sort_values("distances")[:top_k_context_to_use]


if __name__ == "__main__":
    question1 = ("The majority of the pictures feature my two young children."
                 "I'm interested in identifying the foods they are consuming in these photos.")
    question2 = "Find all the photos with both children together. Also elaborate what activities they are doing."

    image_embeddings_path = "outputs/captioned_photos_with_embeddings.csv"
    image_query = ImageQuery(image_embeddings_path)
    answer1 = image_query.answer_question(question1)
    answer2 = image_query.answer_question(question2)
    answer1_simple_prompt = image_query.answer_question(question1, simple_format=True)
    answer2_simple_prompt = image_query.answer_question(question2, simple_format=True)
    answers_path = "outputs/answers.text"

    all_answers = [f"Question one - {question1}\n",
                   f"Question two - {question2}\n",
                   "\n\nCustom answers\n\n",
                   "\n\nAnswer one:", answer1,
                   "\n\nAnswer two:", answer2,
                   "\n\nBasic answers:\n\n",
                   "\n\nAnswer one:", answer1_simple_prompt,
                   "\n\nAnswer two:", answer2_simple_prompt]

    with open(answers_path, 'w', encoding='utf-8') as f:
        for answer in all_answers:
            f.write(answer)
