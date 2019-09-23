from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import errno
import os
from model.VAEGAN import VAEGAN
from algo.vaegan_trainer import VAEGANTrainer
import dateutil
import dateutil.tz
import datetime

if __name__ == "__main__":

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    root_log_dir = "/home/vikas/Documents/vaegan/logs/"
    root_checkpoint_dir = "/home/vikas/Documents/vaegan/ckpt/"
    root_gen_image_save_dir = "/home/vikas/Documents/vaegan/gen_images/"
    batch_size = 64
    updates_per_epoch = 1400
    max_epoch = 250
    z_len = 128           # dimension of sampled noise

    exp_name = "vaegan_%s" % timestamp

    log_dir = os.path.join(root_log_dir, exp_name)
    checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)
    img_save_dir = os.path.join(root_gen_image_save_dir, exp_name)

    os.makedirs(log_dir)
    os.makedirs(checkpoint_dir)
    os.makedirs(img_save_dir)

    model = VAEGAN(
        z_len=z_len,
    )

    algo = VAEGANTrainer(
        model=model,
        batch_size=batch_size,
        z_len=z_len,
        exp_name=exp_name,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        gen_img_dir=img_save_dir,
        max_epoch=max_epoch,
        updates_per_epoch=updates_per_epoch,
        kl_loss_coeff=1/(batch_size*z_len),
        recon_loss_coeff_enc=1/(8*8*256),
        recon_loss_coeff_gen=1e-1/(8*8*256),
        encoder_learning_rate=1e-4,
        generator_learning_rate=1e-4,
        discriminator_learning_rate=1e-4,
    )

    algo.train()

