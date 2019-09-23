from __future__ import print_function
from __future__ import absolute_import

import os
from model.VAEGAN import VAEGAN
from algo.vaegan_tester import VAEGANTester
import dateutil
import dateutil.tz
import datetime

if __name__ == "__main__":

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    root_checkpoint_dir = "/home/vikas/Documents/vaegan/ckpt/vaegan_2019_06_12_10_09_37"
    ckpt_file = "vaegan_2019_06_12_10_09_37_45000.ckpt"
    root_gen_image_save_dir = "/home/vikas/Documents/vaegan/gen_images/vaegan_2019_06_12_10_09_37"
    batch_size = 100
    z_len = 128           # dimension if sampled noise

    exp_name = "test_%s" % timestamp

    img_save_dir = os.path.join(root_gen_image_save_dir, exp_name)
    checkpoint = os.path.join(root_checkpoint_dir, ckpt_file)

    os.makedirs(img_save_dir)

    model = VAEGAN(
        z_len=z_len,
    )

    algo = VAEGANTester(
        model,
        batch_size=batch_size,
        z_len=z_len,
        checkpoint=checkpoint,
        test_img_dir=img_save_dir,
    )

    algo.test()

