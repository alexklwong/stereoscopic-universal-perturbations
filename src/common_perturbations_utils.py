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

from PIL import Image, ImageEnhance
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import ctypes
import numpy as np
import cv2

'''
Initializing Motion Blur class

Copied the implementation from - https://github.com/hendrycks/robustness
'''
# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle

# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def gaussian_blur(img, ksize, stdev):
    '''
    Applies gaussian filter to numpy array

    Arg(s):
        img : numpy[float32]
            array (H x W x C)
        ksize : int
            size of filter
        stdev : float
            standard deviation
    Returns:
        numpy[float32] : array (H x W x C)
    '''

    blur = cv2.GaussianBlur(img, (ksize, ksize), stdev)
    return blur


def jpeg_compression(img):
    '''
    Encodes numpy array as jpeg compression and decodes it back to numpy array

    Arg(s):
        img : numpy[float32]
            H x W x C array
    Returns:
        numpy[float32] : H x W x C array
    '''

    img_arr = (img * 255).astype(np.uint8)

    # Convert RGB image to BGR
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

    # JPEG Compression
    ret, img_bgr = cv2.imencode('.jpeg', img_bgr)
    if not ret:
        raise ValueError('Error occurred during jpeg compression')

    # Decode the jpeg image as numpy array
    img_bgr = cv2.imdecode(img_bgr, cv2.IMREAD_COLOR)

    # Convert BGR to RGB
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img = img / 255

    return img

def quantization(img):
    '''
    Applies quantization to numpy array

    Args:
        img : numpy[float32]
            H x W x C array
    Returns:
        numpy[float32] : H x W x C array
    '''

    img = (img * 255).astype(np.uint8)
    img = img / 255
    return img

def random_brightness(img):
    '''
    Changes brightness by plus minus 20%

    Arg(s):
        img : numpy[float32]
            H x W x C array
    Returns:
        numpy[float32] : H x W x C array
    '''

    data = Image.fromarray((img * 255).astype(np.uint8))
    enhancer = ImageEnhance.Brightness(data)
    factor = random.random() * 0.4 + 0.8
    img_output = enhancer.enhance(factor)
    img = np.asarray(img_output)
    img = img / 255

    return img

def random_contrast(img):
    '''
    Changes contrast by plus minus 20%

    Arg(s):
        img : numpy[float32]
            H x W x C array
    Returns:
        numpy[float32] : H x W x C array
    '''

    data = Image.fromarray((img * 255).astype(np.uint8))
    enhancer = ImageEnhance.Contrast(data)
    factor = random.random() * 0.4 + 0.8
    img_output = enhancer.enhance(factor)
    img = np.asarray(img_output)
    img = img / 255

    return img

def gaussian_noise(x, severity=1):
    '''
    Applies gaussian noise to the image
    https://github.com/hendrycks/robustness

    Arg(s):
        x : numpy[float32]
            H x W x C array
        severity : int
            Level of severity

    Returns:
        numpy[float32] : H x W x C array
    '''

    c = [0.01, 0.04, 0.06, .08, .09, .10][severity - 1]

    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1)

def shot_noise(x, severity=1):
    '''
    Applies shot noise to the image
    https://github.com/hendrycks/robustness

    Arg(s):
        x : numpy[float32]
            H x W x C array
        severity : int
            Level of severity

    Returns:
        numpy[float32] : H x W x C array
    '''

    c = [500, 250, 100, 75, 50][severity - 1]

    return np.clip(np.random.poisson(x * c) / c, 0, 1)

def pixelate(x, severity=1):
    '''
    Pixelates the image
    https://github.com/hendrycks/robustness

    Arg(s):
        x : numpy[float32]
            H x W x C array
        severity : int
            Level of severity

    Returns:
        numpy[float32] : H x W x C array
    '''

    c = [0.5][severity - 1]

    x = Image.fromarray((x * 255).astype(np.uint8))

    x = x.resize((int(640 * c), int(256 * c)), Image.BOX)
    x = x.resize((640, 256), Image.BOX)

    x = np.asarray(x)
    x = x / 255

    return x

def disk(radius, alias_blur=0.1, dtype=np.float32):
    '''
    Helper function for defocus blur
    https://github.com/hendrycks/robustness

    '''

    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

def defocus_blur(x, severity=5):
    '''
    Applies defocus blur to the image
    https://github.com/hendrycks/robustness

    Arg(s):
        x : numpy[float32]
            H x W x C array
        severity : int
            Level of severity

    Returns:
        numpy[float32] : H x W x C array
    '''

    c = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)][severity - 1]

    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x32x32 -> 32x32x3

    return np.clip(channels, 0, 1)

def motion_blur(x, severity=1):
    '''
    Applies motion blur to the image
    https://github.com/hendrycks/robustness

    Arg(s):
        x : numpy[float32]
            H x W x C array
        severity : int
            Level of severity

    Returns:
        numpy[float32] : H x W x C array
    '''

    c = [(6, 1), (6, 1.5), (6, 2), (8, 2), (9, 2.5)][severity - 1]

    x = Image.fromarray((x * 255).astype(np.uint8))

    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                     cv2.IMREAD_UNCHANGED)

    x = np.clip(x[..., [2, 1, 0]], 0, 255) / 255

    return x
