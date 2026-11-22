import os

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None


def _oriented_size(im):
    w, h = im.size
    orientation = im.getexif().get(274, 1)
    return (h, w) if orientation in (5, 6, 7, 8) else (w, h)


def _source_crop_box(x, y, crop, w, h, orientation):
    if orientation == 2:
        return w - x - crop, y, w - x, y + crop
    if orientation == 3:
        return w - x - crop, h - y - crop, w - x, h - y
    if orientation == 4:
        return x, h - y - crop, x + crop, h - y
    if orientation == 5:
        return y, x, y + crop, x + crop
    if orientation == 6:
        return y, h - x - crop, y + crop, h - x
    if orientation == 7:
        return w - y - crop, h - x - crop, w - y, h - x
    if orientation == 8:
        return w - y - crop, x, w - y, x + crop
    return x, y, x + crop, y + crop


def train_files(paths, crop, exclude=()):
    if isinstance(paths, str):
        paths = paths.split(',')
    exclude = tuple(exclude)
    result = []
    for path in paths:
        path = path.strip()
        if not path:
            continue
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames.sort()
            for name in sorted(filenames):
                if not name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                filename = os.path.join(dirpath, name)
                if any(token in filename for token in exclude):
                    continue
                with Image.open(filename) as im:
                    w, h = _oriented_size(im)
                if h >= crop and w >= crop:
                    result.append(filename)
    if not result:
        raise RuntimeError(f'no training images of at least {crop}x{crop} in {paths}')
    return result


def random_crop(path, crop):
    import torch
    with Image.open(path) as im:
        w, h = _oriented_size(im)
        y, x = np.random.randint(h - crop + 1), np.random.randint(w - crop + 1)
        box = _source_crop_box(x, y, crop, *im.size, im.getexif().get(274, 1))
        im = ImageOps.exif_transpose(im.crop(box)).convert('RGB')
        a = np.asarray(im, dtype=np.float32) / 255
    return torch.from_numpy(a).permute(2, 0, 1)
