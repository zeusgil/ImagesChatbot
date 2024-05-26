import os
import openai
import pandas as pd

TEXT_EMBEDDING_LARGE_MODEL = "text-embedding-3-large"


class ImageEmbeddingsCreator:
    """
    Class to create image embeddings using OpenAI API
    """

    def __init__(self, captioned_photos_path: str):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.captioned_photos_path = pd.read_csv(captioned_photos_path)

    def create_image_embeddings(self) -> pd.DataFrame:
        df = self.captioned_photos_path
        batch_size = 10
        embeddings = []
        for i in range(0, len(df), batch_size):
            # Send text data to OpenAI model to get embeddings
            response = openai.Embedding.create(
                input=df.iloc[i:i + batch_size]["photo_caption"].tolist(),
                engine=TEXT_EMBEDDING_LARGE_MODEL
            )

            # Add embeddings to list
            embeddings.extend([data["embedding"] for data in response["data"]])

        # Add embeddings list to dataframe
        df["embeddings"] = embeddings
        return df


if __name__ == "__main__":
    captioned_photos_path = "outputs/photos_captions.csv"
    image_embeddings_creator = ImageEmbeddingsCreator(captioned_photos_path)
    captioned_photos_df = image_embeddings_creator.create_image_embeddings()
    captioned_photos_df.to_csv("outputs/captioned_photos_with_embeddings.csv", index=False)
    print(
        f"Succesfully created image embeddings for {len(captioned_photos_df)} photos.\n Saved to outputs/captioned_photos_with_embeddings.csv.")
