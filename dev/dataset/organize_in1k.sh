cd /home/kergolu/projects/ijepa-parent/ijepa/datasets/in1k/train/ILSVRC2012_img_train

for f in *.tar; do
    # Create directory with tar filename (without extension)
    d="${f%.tar}"
    mkdir -p "$d"
    # Extract tar into that directory
    tar -xf "$f" -C "$d"
    # Remove tar file to save space (optional)
    rm "$f"
done
