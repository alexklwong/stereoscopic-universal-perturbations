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

PRETRAINED_PSMNET_ZIP_URL = \
    GOOGLE_DRIVE_BASE_URL.format('108z1Pp8_AARgFB2MJ3I0gGLUEBA9m_vg')
PRETRAINED_PSMNET_DEFORM6_ZIP_URL = \
    GOOGLE_DRIVE_BASE_URL.format('1eWcusdX3KPZzDiptvI8l9HG51W1Pv_hM')
PRETRAINED_PSMNET_DEFORM25_ZIP_URL = \
    GOOGLE_DRIVE_BASE_URL.format('1PooPzrBCcSNMXXOqemy33wzE0ucOJ0v9')
PRETRAINED_PSMNET_DEFORM25_PATCHMATCH_ZIP_URL = \
    GOOGLE_DRIVE_BASE_URL.format('13EhbBLZGWpbdntP2nIjXGRn94zf48eyg')

PRETRAINED_MODELS_DIRPATH = 'pretrained_stereo_models'
PRETRAINED_MODELS_ZIP_DIRPATH = os.path.join(PRETRAINED_MODELS_DIRPATH, 'zips')

PRETRAINED_PSMNET_ZIP_PATH = \
    os.path.join(PRETRAINED_MODELS_ZIP_DIRPATH, 'pretrained_psmnet.zip')
PRETRAINED_PSMNET_DEFORM6_ZIP_PATH = \
    os.path.join(PRETRAINED_MODELS_ZIP_DIRPATH, 'pretrained_psmnet_deform6.zip')
PRETRAINED_PSMNET_DEFORM25_ZIP_PATH = \
    os.path.join(PRETRAINED_MODELS_ZIP_DIRPATH, 'pretrained_psmnet_deform25.zip')
PRETRAINED_PSMNET_DEFORM25_PATCHMATCH_ZIP_PATH = \
    os.path.join(PRETRAINED_MODELS_ZIP_DIRPATH, 'pretrained_psmnet_deform25_patchmatch.zip')

PRETRAINED_PSMNET_ZIP_PATHS = [
    PRETRAINED_PSMNET_ZIP_PATH,
    PRETRAINED_PSMNET_DEFORM6_ZIP_PATH,
    PRETRAINED_PSMNET_DEFORM25_ZIP_PATH,
    PRETRAINED_PSMNET_DEFORM25_PATCHMATCH_ZIP_PATH
]

PRETRAINED_PSMNET_ZIP_URLS = [
    PRETRAINED_PSMNET_ZIP_URL,
    PRETRAINED_PSMNET_DEFORM6_ZIP_URL,
    PRETRAINED_PSMNET_DEFORM25_ZIP_URL,
    PRETRAINED_PSMNET_DEFORM25_PATCHMATCH_ZIP_URL
]

'''
Downloading and unzipping pretrained models
'''
# Create directories to store pretrained models and zips
for dirpath in [PRETRAINED_MODELS_DIRPATH, PRETRAINED_MODELS_ZIP_DIRPATH]:
    os.makedirs(dirpath, exist_ok=True)

for path, url in zip(PRETRAINED_PSMNET_ZIP_PATHS, PRETRAINED_PSMNET_ZIP_URLS):

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
