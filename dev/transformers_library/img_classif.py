from transformers import AutoImageProcessor, IJepaForImageClassification
import torch
from datasets import load_dataset
"""
Inference using transformers, and datasets library.
This model also needs to be trained.
"""

dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("facebook/ijepa_vith14_1k")
model = IJepaForImageClassification.from_pretrained("facebook/ijepa_vith14_1k")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])