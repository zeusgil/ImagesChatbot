# Custom chatbot_images_project

## Project Description

This project showcases the creation of a searchable photo dataset utilizing OpenAI's GPT models. It involves generating captions and embeddings for a collection of photos, enabling complex queries based on the images' content.

## Files and Usage

### Data Preparation Files
- `create_image_captioning.py`: Script to generate captions for each photo in the dataset.
- `create_image_embeddings.py`: Script to generate embeddings for the photo captions.

### Main Jupyter Notebook
- chatbot_images_project.ipynb: Demonstrates the querying process for the photo dataset.
- Includes steps from setting up the environment and generating captions and embeddings, to querying the dataset.

### Environment Configuration
- `env_variables.txt`: Stores necessary environment variables, including the OpenAI API key.
- Please set up the API key in this file before running the scripts !!!

### Output Files
- `outputs/photos_captions.csv`: Captions for each photo.
- `outputs/captioned_photos_with_embeddings.csv`: Captions and their corresponding embeddings.

## Steps to Use

1. **Environment Setup**
   - Ensure Python and Jupyter are installed.
   - Install required Python packages as listed in the notebook.
   - Set up `env_variables.txt` with your OpenAI API key.

2. **Data Preparation**
   - Place your photos in the `resources/photos` directory.
   - Run `create_image_captioning.py` to generate captions.
   - Run `create_image_embeddings.py` to generate embeddings.

3. **Querying the Dataset**
   - Utilize the custom query class defined in the Jupyter notebook to search the dataset based on various questions about the photo contents.

4. **Performance Demonstration**
   - Follow the examples in the notebook to understand how the system retrieves relevant photos based on custom queries.

## Important Notes

- The effectiveness of queries is dependent on the quality of photo captions and embeddings.
- Further customization of the querying process can improve the relevance of search results.

## Future Work

- Investigate advanced models for more accurate captioning and embedding generation.
- Could try embedd the photos directly and compare the results.
- Implement a user-friendly interface for querying the photo dataset.

## Conclusion

We demonstrated the creation of a searchable photo dataset using OpenAI's GPT models. By generating captions and embeddings for photos, 
we enabled complex queries based on the images' content. This project opens up possibilities for efficient image retrieval systems in various applications.
We also have seen that in our case, RAG stands out as a better method for for the task of image retrieval rather than depending solely on the longer context of the GPT model.