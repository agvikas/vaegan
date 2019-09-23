import os
import glob
import cv2
import numpy
from random import shuffle
from skimage import filters, transform

def _resize(img):
    rescale_size = 64
    bbox = (40, 218-30, 15, 178-15)
    img = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    scale = img.shape[0] / float(rescale_size)
    sigma = numpy.sqrt(scale) / 2.0
    img = filters.gaussian(img, sigma=sigma, multichannel=True)
    img = transform.resize(img, (rescale_size, rescale_size, 3), order=3, mode="constant")
    img = (img * 255).astype(numpy.uint8)
    return img

class data_loader:

    def __init__(self,
                 batch_size,
                 train=True,
                 data_dir="/home/vikas/Documents/ganst/data/portrait_photo",
                 data_ext=".jpg",
                 ):
        self.data_dir = data_dir
        self.data_ext = data_ext
        self.batch_size = batch_size
        self.dirs = [name for name in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, name))]
        self.images = []
        self.pointer = 0
        self.train = train

    def load_data(self):

        path = glob.glob(str(self.data_dir + '/*' + self.data_ext))
        path.sort()
        if self.train:
            for im_path in path[:int(0.9*len(path))]:
                image = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
                image = _resize(image)
                self.images.append(image)

            print("No of images loaded:{}".format(len(self.images)))
        else:
            for im_path in path[int(-0.001*len(path)):]:
                image = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
                image = _resize(image)
                self.images.append(image)

            print("No of test_images loaded:{}".format(len(self.images)))

    def next_batch(self):

        if (self.pointer + self.batch_size) > len(self.images):
            shuffle(self.images)
            self.pointer = 0

        real_images = self.images[self.pointer:self.pointer + self.batch_size]
        self.pointer += self.batch_size

        return real_images

