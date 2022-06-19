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
Define URLs and paths to download pretrained Stereoscopic Universal Perturbations (SUPs)
'''
GOOGLE_DRIVE_BASE_URL = 'https://drive.google.com/uc?id={}'

PRETRAINED_PERTURBATIONS_ZIP_URL = \
    GOOGLE_DRIVE_BASE_URL.format('1owtxBsbL-AEJzzet0CcwofA1vnqRSVwZ')

PRETRAINED_PERTURBATIONS_DIRPATH = 'pretrained_perturbations'
PRETRAINED_PERTURBATIONS_ZIP_DIRPATH = os.path.join(PRETRAINED_PERTURBATIONS_DIRPATH, 'zips')

PRETRAINED_PERTURBATIONS_ZIP_PATH = \
    os.path.join(PRETRAINED_PERTURBATIONS_ZIP_DIRPATH, 'pretrained_perturbations.zip')

PRETRAINED_PERTURBATIONS_ZIP_PATHS = [
    PRETRAINED_PERTURBATIONS_ZIP_PATH
]

PRETRAINED_PERTURBATIONS_ZIP_URLS = [
    PRETRAINED_PERTURBATIONS_ZIP_URL
]

'''
Downloading and unzipping pretrained SUPs
'''
# Create directories to store pretrained models and zips
for dirpath in [PRETRAINED_PERTURBATIONS_DIRPATH, PRETRAINED_PERTURBATIONS_ZIP_DIRPATH]:
    os.makedirs(dirpath, exist_ok=True)

for path, url in zip(PRETRAINED_PERTURBATIONS_ZIP_PATHS, PRETRAINED_PERTURBATIONS_ZIP_URLS):

    filename = os.path.basename(path)

    # Download the zip file if we don't already have it
    if not os.path.exists(path):
        print('Downloading {} to {}'.format(filename, path))
        gdown.download(url, path, quiet=False)
    else:
        print('Found {} at {}'.format(filename, path))

    # Unzip the zip file
    print('Unzipping to {}'.format(PRETRAINED_PERTURBATIONS_DIRPATH))
    with zipfile.ZipFile(path, 'r') as z:
        z.extractall(PRETRAINED_PERTURBATIONS_DIRPATH)
