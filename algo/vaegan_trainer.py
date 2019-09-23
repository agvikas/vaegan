import tensorflow as tf
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
from misc.data_loader import data_loader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

TINY = 1e-8


class VAEGANTrainer(object):
    def __init__(self,
                 model,
                 batch_size,
                 z_len,
                 exp_name="experiment",
                 log_dir="logs",
                 checkpoint_dir="ckt",
                 gen_img_dir="gen_images",
                 max_epoch=100,
                 updates_per_epoch=100,
                 snapshot_interval=5000,
                 margin=0.35,
                 equilibrium=0.68,
                 kl_loss_coeff=1.0,
                 recon_loss_coeff_enc=1.0,
                 recon_loss_coeff_gen=1.0,
                 discriminator_learning_rate=2e-4,
                 encoder_learning_rate=2e-4,
                 generator_learning_rate=2e-4,
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.batch_size = batch_size
        self.z_len = z_len
        self.max_epoch = max_epoch
        self.exp_name = exp_name
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.gen_img_dir = gen_img_dir
        self.snapshot_interval = snapshot_interval
        self.updates_per_epoch = updates_per_epoch
        self.margin = margin
        self.equilibrium = equilibrium
        self.kl_loss_coeff = kl_loss_coeff
        self.recon_loss_coeff_enc = recon_loss_coeff_enc
        self.recon_loss_coeff_gen = recon_loss_coeff_gen
        self.encoder_learning_rate = encoder_learning_rate
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.discriminator_trainer = None
        self.encoder_trainer = None
        self.generator_trainer = None
        self.input_tensor = None
        self.log_vars = []

    def init_opt(self):
        self.input_tensor = input_tensor = tf.placeholder(tf.float32, [self.batch_size, 64, 64, 3])

        z_var = self.sample_prior(self.batch_size, self.z_len)
        encoded, mu, logvar = self.model.encoder(self.input_tensor)

        self.fake_x = self.model.generator(z_var)
        real_d, real_through_d = self.model.discriminator(input_tensor)
        fake_d, _ = self.model.discriminator(self.fake_x, reuse=True)

        self.fake_x_enc = self.model.generator(encoded, reuse=True)
        fake_d_enc, enc_through_d = self.model.discriminator(self.fake_x_enc, reuse=True)

        self.real_cost = - tf.reduce_mean(tf.log(real_d + TINY))
        self.fake_cost = -0.5 * tf.reduce_mean(tf.log(1. - fake_d + TINY) + tf.log(1. - fake_d_enc + TINY))

        encoder_loss = tf.constant(0.0)
        generator_loss = -0.5 * (1 - self.recon_loss_coeff_gen) * \
                         tf.reduce_mean(tf.log(fake_d + TINY) + tf.log(fake_d_enc + TINY))
        discriminator_loss = self.real_cost + self.fake_cost

        self.log_vars.append(("discriminator_loss", discriminator_loss))
        self.log_vars.append(("generator_loss", generator_loss))

        # kl loss
        kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))
        encoder_loss += kl_loss * self.kl_loss_coeff

        # reconstruction loss (fwe)
        recon_loss = - tf.reduce_mean(self.gaussianlogdensity(real_through_d, enc_through_d,
                                                              tf.zeros_like(enc_through_d)))
        '''recon_loss = tf.reduce_mean(tf.reduce_sum(0.5*tf.square(real_through_d-enc_through_d), axis=[1,2,3]))'''
        encoder_loss += recon_loss * self.recon_loss_coeff_enc
        generator_loss += recon_loss * self.recon_loss_coeff_gen

        self.log_vars.append(("kl_loss", kl_loss))
        self.log_vars.append(("reconstruction_loss", recon_loss))

        discriminator_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        encoder_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        generator_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        print("\nNo of trainable discriminator variables: {}".format(len(discriminator_vars)))
        print("No of trainable encoder variables: {}\n".format(len(encoder_vars)))
        print("No of trainable generator variables: {}\n".format(len(generator_vars)))

        self.log_vars.append(("max_real_d", tf.reduce_max(real_d)))
        self.log_vars.append(("min_real_d", tf.reduce_min(real_d)))
        self.log_vars.append(("max_fake_d", tf.reduce_max(fake_d)))
        self.log_vars.append(("min_fake_d", tf.reduce_min(fake_d)))

        global_step = tf.Variable(0, trainable=False)
        self.inc = tf.assign_add(global_step, 1, name='global_step_increment')
        decayed_lr = tf.train.exponential_decay(self.discriminator_learning_rate, global_step, 2800, 0.75,
                                                staircase=True)
        new_lr = tf.cond(decayed_lr > 9e-7, lambda: decayed_lr, lambda: 9e-7)

        self.log_vars.append(("learning_rate", new_lr))

        encoder_optimizer = tf.train.RMSPropOptimizer(new_lr)
        self.encoder_trainer = encoder_optimizer.minimize(loss=encoder_loss, var_list=encoder_vars)

        generator_optimizer = tf.train.RMSPropOptimizer(new_lr)
        self.generator_trainer = generator_optimizer.minimize(loss=generator_loss, var_list=generator_vars)

        discriminator_optimizer = tf.train.RMSPropOptimizer(new_lr)
        self.discriminator_trainer = discriminator_optimizer.minimize(loss=discriminator_loss,
                                                                      var_list=discriminator_vars)

        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

    def train(self):

        self.init_opt()

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)

            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            saver = tf.train.Saver(max_to_keep=4)

            data = data_loader(self.batch_size)
            data.load_data()

            val_imgs = data.images[:self.batch_size]

            counter = 0

            log_vars = [x for _, x in self.log_vars]
            log_keys = [x for x, _ in self.log_vars]

            for epoch in range(self.max_epoch):
                widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                pbar = ProgressBar(maxval=self.updates_per_epoch, widgets=widgets)
                pbar.start()

                all_log_vals = []
                for i in range(self.updates_per_epoch):
                    pbar.update(i)
                    x = data.next_batch()
                    x = np.asarray(x, dtype=np.float32)
                    x = (x / 127.5) - 1.
                    feed_dict = {self.input_tensor: x, self.model.mode: True}

                    dis_train = True
                    gen_train = True
                    real_cost, fake_cost, _ = sess.run([self.real_cost, self.fake_cost, self.inc], feed_dict)
                    if real_cost < self.equilibrium-self.margin or fake_cost < self.equilibrium-self.margin:
                        dis_train = False
                    if real_cost > self.equilibrium+self.margin or fake_cost > self.equilibrium+self.margin:
                        gen_train = False
                    if not (dis_train or gen_train):
                        dis_train = True
                        gen_train = True

                    log_vals = sess.run([self.encoder_trainer] + log_vars, feed_dict)[1:]
                    all_log_vals.append(log_vals)
                    if gen_train:
                        sess.run(self.generator_trainer, feed_dict)
                    if dis_train:
                        sess.run(self.discriminator_trainer, feed_dict)

                    counter += 1

                    if counter % self.snapshot_interval == 0:
                        snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                        fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name),
                                        write_meta_graph=False)
                        print("Model saved in file: %s" % fn)

                x = np.asarray(val_imgs, dtype=np.float32)
                x = (x / 127.5) - 1.

                feed_dict = {self.input_tensor: x, self.model.mode: False}
                summary_str, gen_images, recon_images = sess.run([summary_op, self.fake_x, self.fake_x_enc], feed_dict)
                self.save_generated_images(self.gen_img_dir, gen_images, epoch)
                self.save_generated_images(self.gen_img_dir, recon_images, epoch, False)
                summary_writer.add_summary(summary_str, counter)

                avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                log_dict = dict(zip(log_keys, avg_log_vals))

                log_line = " ".join("%s: %s\n" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals))
                print("Epoch %d | " % (epoch) + log_line)
                sys.stdout.flush()
                if np.any(np.isnan(avg_log_vals)):
                    raise ValueError("NaN detected!")

    def sample_prior(self, batch_size, dim, mu=0., sig=1.):
        mean = mu + tf.zeros([batch_size, dim])
        stddev = sig * tf.ones([batch_size, dim])
        epsilon = tf.random_normal(tf.shape(mean))

        return epsilon * stddev + mean

    def gaussianlogdensity(self, x, mu, logvar, name='GaussianLogDensity'):
        c = np.log(2 * np.pi)
        var = tf.exp(logvar)
        x_mu2 = tf.square(tf.subtract(x, mu))  # [Issue] not sure the dim works or not?
        x_mu2_over_var = tf.div(x_mu2, var + 1e-8)
        log_prob = -0.5 * (c + logvar + x_mu2_over_var)
        log_prob = tf.reduce_sum(log_prob, [1, 2, 3], name=name)  # keep_dims=True,
        return log_prob

    def save_generated_images(self, save_dir, generated_images, epoch, gen=True):

        plt.figure(figsize=(8, 8), num=2)
        gs1 = gridspec.GridSpec(8, 8)
        gs1.update(wspace=0, hspace=0)

        for i in range(64):
            ax1 = plt.subplot(gs1[i])
            ax1.set_aspect('equal')
            image = (generated_images[i, :, :, :] + 1) * 127.5
            fig = plt.imshow(image.astype(np.uint8))
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

        plt.tight_layout()

        if gen:
            img_name = save_dir + '/Sampled_epoch-' + str(epoch) + '.png'
        else:
            img_name = save_dir + '/Reconstructed_epoch-' + str(epoch) + '.png'

        plt.savefig(img_name, bbox_inches='tight', pad_inches=0)
