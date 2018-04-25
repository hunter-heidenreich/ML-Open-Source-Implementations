#!/usr/bin/env python3
"""
DCGAN -> Based off of the paper: https://arxiv.org/pdf/1511.06434.pdf

This is an open source implementation of the DCGAN paper with several modifications.
- Conv2DTranspose is not used because it was not producing results like outlined in the paper
- Upsampling is used instead
- Sizes for convolutions are also shifted due to lack of perfomance and searching for better hyperparameters
- Learning rates have been adjusted as recommended in other sources
- LeakyReLU is used for basically all activations since it had better performance

That should be about it.
I don't think this model has the best performance, but I may re-visit it later.
If you have questions/tips/recommendations/thoughts, contact me via Twitter or email
"""
from keras.models import Model
from keras.layers import Input, BatchNormalization
from keras.layers.core import Dense, Reshape, Activation, Flatten
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import generic_utils
from keras.optimizers import Adam

import numpy as np

import argparse

from data import *

from scipy.misc import imsave


def generator(noise_input, output_shape=(32, 32, 1)):
	"""
	Returns the generator network
	:param noise_input: A shape vector for noise input
	:param output_shape: What output shape we want
	:return: The generator model
	"""
	g_input = Input(shape=noise_input)
	x = Dense(output_shape[0] // 16 * output_shape[1] // 16 * 1024)(g_input)
	x = Reshape((output_shape[0] // 16, output_shape[1] // 16, 1024))(x)
	x = BatchNormalization(momentum=0.5)(x)

	# Conv 1
	x = UpSampling2D()(x)
	x = Conv2D(512, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
	x = LeakyReLU(0.2)(x)
	x = BatchNormalization(momentum=0.5)(x)

	# Conv 2
	x = UpSampling2D()(x)
	x = Conv2D(256, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
	x = LeakyReLU(0.2)(x)
	x = BatchNormalization(momentum=0.5)(x)

	# Conv 3
	x = UpSampling2D()(x)
	x = Conv2D(128, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
	x = LeakyReLU(0.2)(x)
	x = BatchNormalization(momentum=0.5)(x)

	# Conv 4
	x = UpSampling2D()(x)
	x = Conv2D(output_shape[2], kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
	x = Activation('tanh')(x)

	generator_model = Model(inputs=[g_input], outputs=[x])
	return generator_model


def discriminator(input_shape=(32, 32, 1)):
	"""
	Returns the discriminator model
	:param input_shape: The input shape to the discriminator
	:return: The discriminator
	"""
	d_input = Input(shape=input_shape)

	# Conv 1 -> # 64 x 64 x 3
	x = Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same')(d_input)
	x = AveragePooling2D(pool_size=(2, 2))(x)
	x = LeakyReLU(0.2)(x)

	# Conv 2
	x = Conv2D(128, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
	x = AveragePooling2D(pool_size=(2, 2))(x)
	x = LeakyReLU(0.2)(x)
	x = BatchNormalization(momentum=0.5)(x)

	# Conv 3
	x = Conv2D(256, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
	x = AveragePooling2D(pool_size=(2, 2))(x)
	x = LeakyReLU(0.2)(x)
	x = BatchNormalization(momentum=0.5)(x)

	# Conv 4
	x = Conv2D(512, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
	x = AveragePooling2D(pool_size=(2, 2))(x)
	x = LeakyReLU(0.2)(x)

	# Flatten
	x = Flatten()(x)
	x = Dense(1)(x)
	x = Activation('sigmoid')(x)

	discriminator_model = Model(inputs=[d_input], outputs=[x])
	return discriminator_model


def gan(gen, disc, noise_dim, img_shape=(28, 28, 1)):
	"""
	The full gan
	:param gen: The component for generation
	:param disc: The component for discrimination
	:param noise_dim: The expected shape of the noise vector
	:param img_shape: The desired image shape
	:return: The full gan
	"""
	noise_input = Input(shape=noise_dim)
	generated_image = gen(noise_input)

	gan_output = disc(generated_image)
	gan_model = Model(inputs=[noise_input], outputs=[gan_output])
	return gan_model


def main(args):
	"""
	The main training function for the gan
	:param args: CLI args
	:return: Nothing
	"""
	IMG_SHAPE = (64, 64, 3)

	if args.dataset == 'mnist':
		IMG_SHAPE = (64, 64, 1)

	x_data = load_data(args.dataset)

	# Setup models
	generator_model = generator((args.noise_dim, ), output_shape=IMG_SHAPE)
	discriminator_model = discriminator(input_shape=IMG_SHAPE)
	gan_model = gan(generator_model, discriminator_model, (args.noise_dim, ), img_shape=IMG_SHAPE)

	# Compile models
	generator_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00015, beta_1=0.5), metrics=None)
	discriminator_model.trainable = False
	gan_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00015, beta_1=0.5), metrics=None)
	discriminator_model.trainable = True
	discriminator_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=None)

	for e in range(args.epochs):
		progress_bar = generic_utils.Progbar(args.batch_per_epoch * args.batch_size)

		batch_counter = 0
		while batch_counter < args.batch_per_epoch:
			x_real = next(generate_batch(x_data, args.batch_size))
			dx_real, dx_gen = get_d_batch(x_real, generator_model, args.batch_size, args.noise_dim)

			# Train Discriminator
			# d_loss_real = discriminator_model.train_on_batch(dx_real, np.ones(dx_real.shape[0]) + np.random.normal(loc=0.95, scale=0.25))
			d_loss_real = discriminator_model.train_on_batch(dx_real, np.zeros(dx_real.shape[0]) + np.random.normal(loc=0.95, scale=0.25))
			d_loss_gen = discriminator_model.train_on_batch(dx_gen, np.zeros(dx_gen.shape[0]) + np.random.normal(loc=0.15, scale=0.15))

			# Train Generator
			x_gen = sample_noise(1.0, args.batch_size, args.noise_dim)
			discriminator_model.trainable = False
			gen_loss = gan_model.train_on_batch(x_gen, np.ones(x_gen.shape[0]))
			discriminator_model.trainable = True

			batch_counter += 1

			progress_bar.add(args.batch_size)

		print("{}/{} epochs completed".format(e, args.epochs))
		save_generated(generator_model, args.batch_size, args.noise_dim, e, image_output=IMG_SHAPE, data_set=args.dataset, model_name='dcgan')
		save_model_weights(generator_model, discriminator_model, gan_model, e, model_name='dcgan', data_set=args.dataset)


def explore_latent_space(args):
	"""
	Generates frames for an interpolation in the latent space
	:param args: CLI args
	:return: Nothing
	"""
	IMG_SHAPE = (64, 64, 3)

	if args.dataset == 'mnist':
		IMG_SHAPE = (64, 64, 1)

	generator_model = generator((args.noise_dim, ), output_shape=IMG_SHAPE)
	generator_model.load_weights(args.load_model)

	start = sample_noise(1.0, 1, args.noise_dim)

	for repeat in range(2):
		end = sample_noise(1.0, 1, args.noise_dim)

		steps = 50

		alpha_values = np.linspace(0, 1, steps)

		for alpha in alpha_values:
			point = start * (1 - alpha) + end * alpha
			gen = generator_model.predict(point)

			gen = gen.reshape(IMG_SHAPE)

			label = '/home/hunter/git/ML-Open-Source-Implementations/Generative-Adversarial-Networks/generated_images/transitions/dcgan_{}_{}.png'.format(args.dataset, int(steps * alpha) + steps * repeat)
			imsave(label, gen)

		start = end


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='A simple GAN')
	parser.add_argument('--epochs', type=int, default=400, help='The number of epochs to train for')
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--batch_per_epoch', type=int, default=200)
	parser.add_argument('--noise_dim', type=int, default=100)
	parser.add_argument('--dataset', type=str, default='mnist')
	parser.add_argument('--load_model', type=str, default=None)
	parser.add_argument('--latent', action='store_true')

	a = parser.parse_args()

	if a.latent:
		explore_latent_space(a)
	else:
		main(a)
