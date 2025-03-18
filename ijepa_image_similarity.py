"""
Image similarity calculator using I-JEPA embeddings.

This script compares two images using embeddings from a pre-trained I-JEPA ViT-H/14 model.
It calculates cosine similarity between the embedding vectors to measure image relatedness.
"""

import requests
from PIL import Image
import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoModel, AutoProcessor


def infer(image: Image.Image) -> torch.Tensor:
    """
    Generate embedding vector for an image using the I-JEPA model.

    Args:
        image: PIL Image to process

    Returns:
        Tensor containing the image embedding
    """
    inputs = processor(image, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def calculate_similarity(img1_url: str, img2_url: str) -> float:
    """
    Calculate similarity between two images from their URLs.

    Args:
        img1_url: URL of the first image
        img2_url: URL of the second image

    Returns:
        Cosine similarity score (range -1 to 1)
    """
    # Load images from URLs
    image_1 = Image.open(requests.get(img1_url, stream=True).raw)
    image_2 = Image.open(requests.get(img2_url, stream=True).raw)

    # Generate embeddings
    embed_1 = infer(image_1)
    embed_2 = infer(image_2)

    # Calculate similarity
    return cosine_similarity(embed_1, embed_2).item()


if __name__ == "__main__":
    # Load pre-trained I-JEPA model
    model_id = "jmtzt/ijepa_vith14_1k"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)

    # Example image URLs from COCO dataset
    url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
    url_2 = "http://images.cocodataset.org/val2017/000000219578.jpg"

    # Calculate and print similarity
    similarity = calculate_similarity(url_1, url_2)
    print(f"Image similarity: {similarity:.4f}")
