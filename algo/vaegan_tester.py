import tensorflow as tf
import numpy as np
from misc.data_loader import data_loader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

TINY = 1e-8


class VAEGANTester(object):
    def __init__(self,
                 model,
                 batch_size=100,
                 z_len=128,
                 checkpoint=None,
                 test_img_dir=None,
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.batch_size = batch_size
        self.z_len = z_len
        self.checkpoint = checkpoint
        self.test_img_dir = test_img_dir
        self.input_tensor = None

    def init_opt(self):
        self.input_tensor = input_tensor = tf.placeholder(tf.float32, [None, 64, 64, 3])

        z_var = self.sample_prior(self.batch_size, self.z_len)

        self.fake_x = self.model.generator(z_var)
        enc_out, _, _ = self.model.encoder(self.input_tensor)
        self.fake_x_enc = self.model.generator(enc_out, reuse=True)

    def test(self):

        self.init_opt()

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)

            saver = tf.train.Saver()
            saver.restore(sess, self.checkpoint)

            data = data_loader(self.batch_size, False)
            data.load_data()

            x = data.images

            #for i in range(int(len(x)/self.batch_size)):
            for i in range(5):
                #y = x[(i*self.batch_size) : (i*self.batch_size) + self.batch_size]
                y = x[(i * 10): (i * 10) + 10] * 10
                y = np.asarray(y, dtype=np.float32)
                y = (y / 127.5) - 1

                feed_dict = {self.input_tensor: y,
                             self.model.mode: False}
                gen_images, tran_images = sess.run([self.fake_x, self.fake_x_enc],
                                                    feed_dict)
                #summary_str, = sess.run([summary_op], feed_dict)
                self.save_val_images(self.test_img_dir, y, i, 'real')
                self.save_val_images(self.test_img_dir, gen_images, i, 'gen')
                self.save_val_images(self.test_img_dir, tran_images, i, 'recon')

    def sample_prior(self, batch_size, dim, mu=0., sig=1.):
        mean = mu + tf.zeros([batch_size, dim])
        stddev = sig * tf.ones([batch_size, dim])
        epsilon = tf.random_normal(tf.shape(mean))

        return mean + epsilon * stddev

    def save_val_images(self, save_dir, generated_images, epoch, gen):

        plt.figure(figsize=(10, 10), num=2)
        gs1 = gridspec.GridSpec(10, 10)
        gs1.update(wspace=0, hspace=0)

        for i in range(100):
            ax1 = plt.subplot(gs1[i])
            ax1.set_aspect('equal')
            image = (generated_images[i, :, :, :] + 1) * 127.5
            fig = plt.imshow(image.astype(np.uint8))
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

        plt.tight_layout()
        if gen=='real':
            img_name = save_dir + '/original-' + str(epoch) + '.png'
        elif gen=='gen':
            img_name = save_dir + '/generated-' + str(epoch) + '.png'
        else:
            img_name = save_dir + '/translated-' + str(epoch) + '.png'

        plt.savefig(img_name, bbox_inches='tight', pad_inches=0)
