'''
Author: Zachery Berger <zackeberger@g.ucla.edu>, Parth Agrawal <parthagrawal24@g.ucla.edu>, Tian Yu Liu <tianyu139@g.ucla.edu>, Alex Wong <alexw@cs.ucla.edu>
If you use this code, please cite the following paper:

Z. Berger, P. Agrawal, T. Liu, S. Soatto, and A. Wong. Stereoscopic Universal Perturbations across Different Architectures and Datasets.
https://arxiv.org/pdf/2112.06116.pdf

@inproceedings{berger2022stereoscopic,
  title={Stereoscopic Universal Perturbations across Different Architectures and Datasets},
  author={Berger, Zachery and Agrawal, Parth and Liu, Tian Yu and Soatto, Stefano and Wong, Alex},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
'''

import os, gdown, zipfile


'''
Define URLs and paths to download pretrained models
'''
GOOGLE_DRIVE_BASE_URL = 'https://drive.google.com/uc?id={}'

PRETRAINED_DEEPPRUNER_ZIP_URL = \
    GOOGLE_DRIVE_BASE_URL.format('1VHsPqmfrZBs2d7IjwzKX4D-2uvhlMLxp')

PRETRAINED_MODELS_DIRPATH = 'pretrained_stereo_models'
PRETRAINED_MODELS_ZIP_DIRPATH = os.path.join(PRETRAINED_MODELS_DIRPATH, 'zips')

PRETRAINED_DEEPPRUNER_ZIP_PATH = \
    os.path.join(PRETRAINED_MODELS_ZIP_DIRPATH, 'pretrained_deeppruner.zip')

PRETRAINED_DEEPPRUNER_ZIP_PATHS = [
    PRETRAINED_DEEPPRUNER_ZIP_PATH
]

PRETRAINED_DEEPPRUNER_ZIP_URLS = [
    PRETRAINED_DEEPPRUNER_ZIP_URL
]

'''
Downloading and unzipping pretrained models
'''
# Create directories to store pretrained models and zips
for dirpath in [PRETRAINED_MODELS_DIRPATH, PRETRAINED_MODELS_ZIP_DIRPATH]:
    os.makedirs(dirpath, exist_ok=True)

for path, url in zip(PRETRAINED_DEEPPRUNER_ZIP_PATHS, PRETRAINED_DEEPPRUNER_ZIP_URLS):

    filename = os.path.basename(path)

    # Download the zip file if we don't already have it
    if not os.path.exists(path):
        print('Downloading {} to {}'.format(filename, path))
        gdown.download(url, path, quiet=False)
    else:
        print('Found {} at {}'.format(filename, path))

    # Unzip the zip file
    print('Unzipping to {}'.format(PRETRAINED_MODELS_DIRPATH))
    with zipfile.ZipFile(path, 'r') as z:
        z.extractall(PRETRAINED_MODELS_DIRPATH)
