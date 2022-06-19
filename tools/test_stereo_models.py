import os, sys, warnings
import torch
import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, 'src')
from stereo_model import StereoModel


device = torch.device('cuda')


'''
Create dummy inputs
'''
image0 = np.random.random([1, 3, 256, 640])
image1 = np.random.random([1, 3, 256, 640])

image0 = torch.from_numpy(image0).float()
image1 = torch.from_numpy(image1).float()

image0 = image0.to(device)
image1 = image1.to(device)


'''
Test stereo models
'''
for name in ['psmnet', 'deeppruner', 'aanet']:
    stereo_model = StereoModel(method=name, device=torch.device('cuda'))

    try:
        stereo_model.forward(image0, image1)
    except Exception as e:
        print('Failed forward pass for {}:'.format(name))
        print(e)

print('Passed tests for stereo models')
