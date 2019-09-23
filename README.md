# VAEGAN

Implementation of [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/abs/1512.09300v2) in tensorflow. The work combines Variation Auto-Encoder(VAE) and
Generative Adversarial Network(GAN), and uses reconstruction error expressed in discriminator instead of pixelwise reconstruction error in VAE.

Refer to official implemetation for further details [here](https://github.com/andersbll/autoencoding_beyond_pixels) 

## Requirements

1) Python 3.5 (not tested on python 2.7)
2) tensorflow (tested on 1.09)
3) openCV
4) scikit-image

## Usage

```pyhton run_vaegan_trainer.py``` for training the network and ```python run_tester.py``` for testing.

Make sure to create "logs", "ckpt" and "gen_images" directory in the project directory.

## Results 

Original images

<img src="https://user-images.githubusercontent.com/38666732/65448303-245ae080-de56-11e9-9702-3b3e0c12250e.png" width=500>

Reconstructions

<img src="https://user-images.githubusercontent.com/38666732/65448312-2755d100-de56-11e9-85a5-e1351f985fd4.png" width=500>

Generated images

<img src="https://user-images.githubusercontent.com/38666732/65448293-1f962c80-de56-11e9-9c01-a6f09800cb56.png" width=500>
