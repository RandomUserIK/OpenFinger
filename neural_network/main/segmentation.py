import os
import sys
import logging
import tensorflow as tf
import mrcnn.model as mrcnn_model
import roi
import cv2

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, 'logs/')
WEIGHTS_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')
DEVICE = '/gpu:0'
MODE = 'inference'


def init_logger():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='\n%(message)s')


class InferenceConfig(roi.RoiConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()


class Segmentation:

    def __init__(self) -> None:
        self.model = None
        self.init_model()

    def init_model(self) -> None:
        logging.info('Initializing the model')
        with tf.device(DEVICE):
            self.model = mrcnn_model.MaskRCNN(mode=MODE, model_dir=MODEL_DIR, config=config)
            self.load_weights()
            print()

    def load_weights(self) -> None:
        logging.info('Loading weights from ' + WEIGHTS_PATH)
        self.model.load_weights(WEIGHTS_PATH, by_name=True)

    def detect(self, image) -> list:
        if image is None:
            logging.warning('An empty image was provided')
            raise ValueError('A invalid value was provided as an image argument')
        return self.model.detect([image], 1)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                            [tf.config.experimental.VirtualDeviceConfiguration(
                                                                memory_limit=8192)])

    init_logger()
    segmentation = Segmentation()
    img = cv2.imread('/home/xkovac/Documents/test.bmp')
    result = segmentation.detect(img)
    for i in range(len(result)):
        print(result[i])
    print(result)
    print()
