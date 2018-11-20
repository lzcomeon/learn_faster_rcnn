
import os

import xml.etree.ElementTree as ET
import numpy as np
from .util import read_image



class VOCBboxDataset(object):
    """Bounding box dataset for PASCAL VOC

    The type of the image, the bounding boxes and the labels are as follows.
    img.dtype == np.float32
    bbox.dtype == np.float32
    label.dtype == np.int32
    difficult.dtype == numpy.bool

    """

    def __init__(self, data_dir, split='trainval', use_difficult=False,
                 return_difficult=False,
                 ):
        id_list_file = os.path.join(data_dir, 'ImageSets/Main/{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_images(self, i):
        """Returns the i-th example

        Returns a color image and bounding boxes. The image is in CHW format.
        Args:
            i(int): The index of the examples
        Returns:
            tuple of an image and bounding boxes
        """
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotation', id_+'.xml'))
        bbox = list()
        label = list()
        for obj in anno.findall('object'):
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                        int(bndbox_anno.find(tag).text) -1
                        for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))

        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_+'.jpg')
        img = read_image(img_file, color=True)

        return img, bbox, label

    __getitem__ = get_images









VOC_BBOX_LABEL_NAMES = ('tomato')

# path = '/home/lz/Lab/pytorch/Faster_rcnn/datasets/tomato/VOCdevkit/'
# imdb = VOCBboxDataset(path)
# imdb.get_images(1)
# print(imdb.__len__())