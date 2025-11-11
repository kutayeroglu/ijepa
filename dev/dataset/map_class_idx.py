# UNUSED: Left in case we need it later
# """Utility script to extract the ImageNet1K index-to-synset mapping.

# This script reads the `meta.mat` file contained in the ImageNet 2012 devkit
# and builds a simple dictionary that maps the 1-based class indices used by
# ImageNet to their corresponding WordNet IDs (WNIDs). The resulting mapping is
# stored in the `idx_to_synset` dictionary for downstream scripts to reuse.
# """

# from scipy.io import loadmat

# meta = loadmat(
#     "/home/kergolu/datasets/in1k/_unzipped/ILSVRC2012_devkit_t12/data/meta.mat",
#     squeeze_me=True,
# )
# synsets = meta["synsets"]  # structured array of length 1000
# idx_to_synset = {}
# for entry in synsets:
#     idx = int(entry[0])  # ILSVRC2012_ID (1..1000)
#     wnid = entry[1]  # WNID string like 'n01440764'
#     idx_to_synset[idx] = wnid
