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
Define URLs and paths to download kitti segmentation maps zip file
'''
GOOGLE_DRIVE_BASE_URL = 'https://drive.google.com/uc?id={}'

KITTI_SEGMENTATION_ZIP_URL = \
    GOOGLE_DRIVE_BASE_URL.format('1sMP18O0u3C96-GjQgzeKG91RDl0Ykuck')

KITTI_SEGMENTATION_DIRPATH = 'data'

KITTI_SEGMENTATION_ZIP_PATH = \
    os.path.join(KITTI_SEGMENTATION_DIRPATH, 'kitti_segmentation.zip')

KITTI_SEGMENTATION_ZIP_PATHS = [
    KITTI_SEGMENTATION_ZIP_PATH
]

KITTI_SEGMENTATION_ZIP_URLS = [
    KITTI_SEGMENTATION_ZIP_URL
]

'''
Downloading and unzipping kitti segmentation maps
'''
# Create directories to store kitti segmentation maps
os.makedirs(KITTI_SEGMENTATION_DIRPATH, exist_ok=True)

for path, url in zip(KITTI_SEGMENTATION_ZIP_PATHS, KITTI_SEGMENTATION_ZIP_URLS):

    filename = os.path.basename(path)

    # Download the zip file if we don't already have it
    if not os.path.exists(path):
        print('Downloading {} to {}'.format(filename, path))
        gdown.download(url, path, quiet=False)
    else:
        print('Found {} at {}'.format(filename, path))

    # Unzip the zip file
    print('Unzipping to {}'.format(KITTI_SEGMENTATION_DIRPATH))
    with zipfile.ZipFile(path, 'r') as z:
        z.extractall(KITTI_SEGMENTATION_DIRPATH)
