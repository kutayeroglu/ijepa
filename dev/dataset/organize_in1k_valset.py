import os
import requests
import shutil

# Download the synset labels
url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt'
response = requests.get(url)
synset_labels = response.text.strip().split('\n')

# Move images
val_images_dir = '/home/kergolu/projects/ijepa-parent/ijepa/datasets/in1k/val/ILSVRC2012_img_val'
output_val_dir = '/home/kergolu/projects/ijepa-parent/ijepa/datasets/in1k/val'

for i, synset in enumerate(synset_labels):
    img_name = f'ILSVRC2012_val_{i+1:08d}.JPEG'
    src = os.path.join(val_images_dir, img_name)
    dst_dir = os.path.join(output_val_dir, synset)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, img_name)
    shutil.move(src, dst)