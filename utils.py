import torch
import random
import numpy as np
import os
import struct
import gzip
from PIL import Image

def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mmd(x, y, gammas, device):
    gammas = gammas.to(device)

    cost = torch.mean(gram_matrix(x, x, gammas=gammas)).to(device)
    cost += torch.mean(gram_matrix(y, y, gammas=gammas)).to(device)
    cost -= 2 * torch.mean(gram_matrix(x, y, gammas=gammas)).to(device)

    if cost < 0:
        return torch.tensor(0).to(device)
    return cost


def gram_matrix(x, y, gammas):
    gammas = gammas.unsqueeze(1)
    pairwise_distances = torch.cdist(x, y, p=2.0)

    pairwise_distances_sq = torch.square(pairwise_distances)
    tmp = torch.matmul(gammas, torch.reshape(pairwise_distances_sq, (1, -1)))
    tmp = torch.reshape(torch.sum(torch.exp(-tmp), 0), pairwise_distances_sq.shape)
    return tmp


def resize_and_crop(img, size=(100,100), crop_type='middle'):
    # If height is higher we resize vertically, if not we resize horizontally
    # Get current and desired ratio for the images
    img_ratio = img.size[0] / float(img.size[1])
    ratio = size[0] / float(size[1])
    # The image is scaled/cropped vertically or horizontally
    # depending on the ratio
    if ratio > img_ratio:
        img = img.resize((
            size[0],
            int(round(size[0] * img.size[1] / img.size[0]))),
            Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, img.size[0], size[1])
        elif crop_type == 'middle':
            box = (
                0,
                int(round((img.size[1] - size[1]) / 2)),
                img.size[0],
                int(round((img.size[1] + size[1]) / 2)))
        elif crop_type == 'bottom':
            box = (0, img.size[1] - size[1], img.size[0], img.size[1])
        else:
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    elif ratio < img_ratio:
        img = img.resize((
            int(round(size[1] * img.size[0] / img.size[1])),
            size[1]),
            Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, size[0], img.size[1])
        elif crop_type == 'middle':
            box = (
                int(round((img.size[0] - size[0]) / 2)),
                0,
                int(round((img.size[0] + size[0]) / 2)),
                img.size[1])
        elif crop_type == 'bottom':
            box = (
                img.size[0] - size[0],
                0,
                img.size[0],
                img.size[1])
        else:
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    else:
        img = img.resize((
            size[0],
            size[1]),
            Image.ANTIALIAS)
    # If the scale is the same, we do not need to crop
    return img


def _load_uint8(f):
    idx_dtype, ndim = struct.unpack('BBBB', f.read(4))[2:]
    shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data


def _save_uint8(data, f):
    data = np.asarray(data, dtype=np.uint8)
    f.write(struct.pack('BBBB', 0, 0, 0x08, data.ndim))
    f.write(struct.pack('>' + 'I' * data.ndim, *data.shape))
    f.write(data.tobytes())


def save_idx(data: np.ndarray, path: str):
    """Writes an array to disk in IDX format.
    Parameters
    ----------
    data : array_like
        Input array of dtype ``uint8`` (will be coerced if different dtype).
    path : str
        Path of the output file. Will compress with `gzip` if path ends in '.gz'.
    References
    ----------
    http://yann.lecun.com/exdb/mnist/
    """
    open_fcn = gzip.open if path.endswith('.gz') else open
    with open_fcn(path, 'wb') as f:
        _save_uint8(data, f)


def load_idx(path: str) -> np.ndarray:
    """Reads an array in IDX format from disk.
    Parameters
    ----------
    path : str
        Path of the input file. Will uncompress with `gzip` if path ends in '.gz'.
    Returns
    -------
    np.ndarray
        Output array of dtype ``uint8``.
    References
    ----------
    http://yann.lecun.com/exdb/mnist/
    """
    open_fcn = gzip.open if path.endswith('.gz') else open
    with open_fcn(path, 'rb') as f:
        return _load_uint8(f)
