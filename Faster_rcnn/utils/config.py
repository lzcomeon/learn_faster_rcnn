from pprint import pprint


# Default Configs for training

class Config(object):
    # data
    voc_data_dir = '/home/lz/Lab/pytorch/Faster_rcnn/datasets/tomato/VOCdevkit/'
    min_size = 600
    max_size = 1000
    batch_szie = 16



    caffe_pretrain = False # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'






    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)  # setattr(x, 'y', v) is equivalent to ``x.y = v''

        print('************use config************')
        pprint(self._state_dict())


    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

opt = Config()