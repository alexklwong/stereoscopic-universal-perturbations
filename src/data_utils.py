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

import cv2, re
import numpy as np
from PIL import Image


def read_paths(filepath):
    '''
    Reads a newline delimited file containing paths

    Arg(s):
        filepath : str
            path to file to be read
    Return:
        list : list of paths
    '''

    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip('\n')
            # If there was nothing to read
            if path == '':
                break
            path_list.append(path)

    return path_list

def write_paths(filepath, paths):
    '''
    Stores line delimited paths into file

    Arg(s):
        filepath : str
            path to file to save paths
        paths : list
            paths to write into file
    '''

    with open(filepath, 'w') as o:
        for idx in range(len(paths)):
            o.write(paths[idx] + '\n')

def load_image(path, normalize=True, data_format='HWC'):
    '''
    Loads an RGB image

    Arg(s):
        path : str
            path to RGB image
        normalize : bool
            if set, then normalize image between [0, 1]
        data_format : str
            'CHW', or 'HWC'
    Returns:
        numpy[float32] : H x W x C or C x H x W image
    '''

    # Load image
    image = Image.open(path).convert('RGB')

    # Convert to numpy
    image = np.asarray(image, np.float32)

    if data_format == 'HWC':
        pass
    elif data_format == 'CHW':
        image = np.transpose(image, (2, 0, 1))
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    # Normalize
    image = image / 255.0 if normalize else np.asarray(image, np.uint8)

    return image

def load_disparity(path, multiplier=256.0, data_format='HWC'):
    '''
    Loads a disparity image

    Arg(s):
        path : str
            path to disparity image
        multiplier : float
            multiplier to convert saved intensities to disparities
        data_format : str
            'CHW', or 'HWC'
    Returns:
        numpy[float32] : H x W x C or C x H x W disparity image
    '''

    # Load image and resize
    disparity = Image.open(path).convert('I')

    # Convert unsigned int16 to disparity values
    disparity = np.asarray(disparity, np.uint16)
    disparity = disparity / multiplier

    if disparity.ndim == 2:
        disparity = np.expand_dims(disparity, axis=-1)

    if data_format == 'HWC':
        pass
    elif data_format == 'CHW':
        disparity = np.transpose(disparity, (2, 0, 1))
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return np.asarray(disparity, np.float32)

def save_disparity(disparity, path, multiplier=256.0):
    '''
    Saves a disparity image

    Arg(s):
        disparity : numpy[float32]
            disparity image
        path : str
            path to disparity image
        multiplier : float
            multiplier to convert saved intensities to disparities
    '''

    # Convert unsigned int16 to disparity values
    disparity = disparity * multiplier
    disparity = np.asarray(disparity, np.uint32)

    disparity = np.squeeze(disparity)

    disparity = Image.fromarray(disparity, mode='I')
    disparity.save(path)

def read_pfm(path):
    '''
    Reads the content of a pfm file

    Arg(s):
        path : str
            path to pfm image
    Returns:
        numpy[float32] : pfm content
    '''

    content = open(path, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = content.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', content.readline().decode("ascii"))

    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(content.readline().decode("ascii").rstrip())

    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(content, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)

    return data

def resize(T, shape, interp_type='lanczos', data_format='HWC'):
    '''
    Resizes a tensor

    Arg(s):
        T : numpy[float32]
            tensor to resize
        shape : tuple[int]
            (height, width) to resize tensor
        interp_type : str
            interpolation for resize
        data_format : str
            'CHW', or 'HWC', 'CDHW', 'DHWC'
    Returns:
        numpy[float32] : image resized to height and width
    '''

    dtype = T.dtype

    if interp_type == 'nearest':
        interp_type = cv2.INTER_NEAREST
    elif interp_type == 'area':
        interp_type = cv2.INTER_AREA
    elif interp_type == 'bilinear':
        interp_type = cv2.INTER_LINEAR
    elif interp_type == 'lanczos':
        interp_type = cv2.INTER_LANCZOS4
    else:
        raise ValueError('Unsupport interpolation type: {}'.format(interp_type))

    if shape is None or any([x is None or x <= 0 for x in shape]):
        return T

    n_height, n_width = shape

    # Resize tensor
    if data_format == 'CHW':
        # Tranpose from CHW to HWC
        R = np.transpose(T, (1, 2, 0))

        # Resize and transpose back to CHW
        R = cv2.resize(R, dsize=(n_width, n_height), interpolation=interp_type)
        R = np.reshape(R, (n_height, n_width, T.shape[0]))
        R = np.transpose(R, (2, 0, 1))

    elif data_format == 'HWC':
        R = cv2.resize(T, dsize=(n_width, n_height), interpolation=interp_type)
        R = np.reshape(R, (n_height, n_width, T.shape[2]))

    else:
        raise ValueError('Unsupport data format: {}'.format(data_format))

    return R.astype(dtype)


def load_calibration(path):
    '''
    Loads the calibration matrices for each camera (KITTI) and stores it as map

    Arg(s):
        path : str
            path to file to be read
    Returns:
        dict[str, float32] : map containing camera intrinsics keyed by camera id
    '''

    float_chars = set("0123456789.e+- ")
    data = {}

    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                try:
                    data[key] = np.asarray(
                        [float(x) for x in value.split(' ')])
                except ValueError:
                    pass
    return data


def save_depth(z, path):
    '''
    Saves a depth map to a 16-bit PNG file

    Arg(s):
        z : numpy[float32]
            depth map
        path : str
            path to store depth map
    '''

    z = np.uint32(z * 256.0)
    z = Image.fromarray(z, mode='I')
    z.save(path)
