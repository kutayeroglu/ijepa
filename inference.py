import requests
from PIL import Image
from torch.nn.functional import cosine_similarity

from transformers import AutoModel, AutoProcessor

url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
url_2 = "http://images.cocodataset.org/val2017/000000219578.jpg"
image_1 = Image.open(requests.get(url_1, stream=True).raw)
image_2 = Image.open(requests.get(url_2, stream=True).raw)

model_id = "jmtzt/ijepa_vith14_1k"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)


def infer(image):
    inputs = processor(image, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


embed_1 = infer(image_1)
embed_2 = infer(image_2)

similarity = cosine_similarity(embed_1, embed_2)
print(similarity)
