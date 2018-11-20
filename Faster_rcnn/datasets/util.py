import numpy as np
from PIL import Image
import random


def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file
    This function reads an image from given file. The image is CHW format and
    the range of its value is math`[0, 255`, if :obj:`color = True`, the
    order of the channels is RGB

    :param path: A path pf image file
    :param dtype:
    :param color:
        if True: the order of the channels is RGB
        if False: this function returns a grayscale image
    :return: numpy.ndarry: An image
    """

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=np.float32)
        # print('img origin shape:', img.shape)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose(2, 0, 1)


def resize_bbox(bbox, in_size, out_size):

    """Resize the bounding boxes according to image resize

    Args:
        bbox ~(numpy.ndarry): An array whose shape is :math:`(R, 4)`.
            `R` is the number of bounding boxes
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized
    Returns:
        ~numpy.ndarry (R, 4):
        Bounding boxes rescaled according to the given image shapes

    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 3] = x_scale * bbox[:, 3]

    return bbox


def random_flip(img, y_random=False, x_random=False,
                return_param=False, copy=False):
    """Randomly flip an image in vertical or horizontal direction

    Args:
        img ~(numpy.ndarry): This is in CHW format
        y_random (bool): Randomly flip in vertical direction
        x_random (boox): Randomly flip in horizontal direction
        return_param (bool): Returns information of flip
        copy (bool):
    """

    y_flip, x_flip =False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img


def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    """Flip bounding boxes accordingly

    Args:
        bbox (np.ndarry): (R, 4)
        size (tuple): A tuple of length 2. The height and the width
            of the image before resized

    Returns:
         np.ndarry: (R, 4)
         Bounding boxes flipped according to the given flips.
    """

    H, W = size
    bbox = bbox.copy()
    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox






if __name__ == '__main__':
    img = read_image('/home/lz/Lab/pytorch/Faster_rcnn/datasets/tomato/VOCdevkit/JPEGImages/001_1.jpg')
    # print(img.shape)