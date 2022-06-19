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

# Perturbation model settings
N_EPOCH                             = 1
N_IMAGE_HEIGHT                      = 256
N_IMAGE_WIDTH                       = 640
OUTPUT_NORM                         = 0.02
WEIGHT_INITIALIZER                  = 'random'
GRADIENT_SCALE                      = 0.0001
ATTACK                              = 'tile'
N_PERTURBATION_HEIGHT               = 64
N_PERTURBATION_WIDTH                = 64

# Optimization settings
N_BATCH                             = 8

# Stereo method settings
STEREO_METHOD_AVAILABLE             = ['aanet', 'aanet_plus', 'deeppruner', 'deeppruner_fast', 'deeppruner_best', 'psmnet']

STEREO_METHOD                       = 'deeppruner'

# Checkpoint settings
N_CHECKPOINT                        = 1000

# Hardware settings
DEVICE                              = 'cuda'
CUDA                                = 'cuda'
CPU                                 = 'cpu'
GPU                                 = 'gpu'
N_WORKER                            = 8

# Defense settings
GAUSSIAN_KSIZE                      = 5
GAUSSIAN_STDEV                      = 1

# Finetune settings
LEARNING_RATES                      = [1e-6, 5e-7, 2e-7]
LEARNING_SCHEDULE                   = [600, 800, 1000]

# Segmentation settings
SEG_LABELS                          = {
                                        0: 'road',
                                        1: 'sidewalk',
                                        2: 'building',
                                        3: 'wall',
                                        4: 'fence',
                                        5: 'pole',
                                        6: 'traffic light',
                                        7: 'traffic sign',
                                        8: 'vegetation',
                                        9: 'terrain',
                                        10: 'sky',
                                        11: 'person',
                                        12: 'rider',
                                        13: 'car',
                                        14: 'truck',
                                        15: 'bus',
                                        16: 'train',
                                        17: 'motorcycle',
                                        18: 'bicycle'
                                    }
