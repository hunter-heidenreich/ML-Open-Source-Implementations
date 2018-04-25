#!/usr/bin/env python3
"""
Original GAN -> Based off of the original GAN paper: https://arxiv.org/pdf/1406.2661.pdf

This is an open source implementation of the original Goodfellow GAN paper

A couple thoughts:
- Not a very reliable model
- I wanted to create this only using dense layers, due to wanting to use only convolutions for the DCGAN. This was a sort of personal challenge.
- No real explicit hyperparameters were mentioned in the paper. These were chosed based off of other sources and experimentation

If you have questions/tips/recommendations/thoughts, contact me via Twitter or email
"""
from keras.models import Model
from keras.layers import Input, BatchNormalization
from keras.layers.core import Dense, Reshape, Activation, Flatten
from keras.utils import generic_utils
from keras.optimizers import Adam

import numpy as np

import argparse

from data import *


def generator(noise_input, output_shape=(32, 32, 1)):
	"""
	Returns the generator network
	:param noise_input: A shape vector for noise input
	:param output_shape: What output shape we want
	:return: The generator model
	"""
	g_input = Input(shape=noise_input)
	x = Dense(256)(g_input)
	x = Activation('relu')(x)
	x = BatchNormalization(momentum=0.8)(x)
	x = Dense(512)(x)
	x = Activation('relu')(x)
	x = BatchNormalization(momentum=0.8)(x)
	x = Dense(1025)(x)
	x = Activation('relu')(x)
	x = BatchNormalization(momentum=0.8)(x)
	x = Dense(output_shape[0] * output_shape[1] * output_shape[2])(x)
	x = Activation('tanh')(x)
	x = Reshape(output_shape)(x)

	generator_model = Model(inputs=[g_input], outputs=[x])
	return generator_model


def discriminator(input_shape=(32, 32, 1)):
	"""
	Returns the discriminator model
	:param input_shape: The input shape to the discriminator
	:return: The discriminator
	"""
	d_input = Input(shape=input_shape)
	x = Flatten()(d_input)
	x = Dense(512)(x)
	x = Activation('relu')(x)
	x = Dense(256)(x)
	x = Activation('relu')(x)
	x = Dense(1)(x)
	x = Activation('sigmoid')(x)

	discriminator_model = Model(inputs=[d_input], outputs=[x])
	return discriminator_model


def gan(gen, disc, noise_dim, img_shape=(32, 32, 1)):
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
	x_data = load_data('mnist')
	print(x_data.shape)

	# Setup models
	generator_model = generator((args.noise_dim, ))
	discriminator_model = discriminator()
	gan_model = gan(generator_model, discriminator_model, (args.noise_dim, ))

	# Compile models
	generator_model.compile(loss='binary_crossentropy', optimizer=Adam())
	discriminator_model.trainable = False
	gan_model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
	discriminator_model.trainable = True
	discriminator_model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

	for e in range(args.epochs):
		progress_bar = generic_utils.Progbar(args.batch_per_epoch * args.batch_size)

		batch_counter = 0
		while batch_counter < args.batch_per_epoch:
			x_real = next(generate_batch(x_data, args.batch_size))
			dx_real, dx_gen = get_d_batch(x_real, generator_model, args.batch_size, args.noise_dim)

			# Train Discriminator
			d_loss_real = discriminator_model.train_on_batch(dx_real, np.ones(dx_real.shape[0]))
			d_loss_gen = discriminator_model.train_on_batch(dx_gen, np.zeros(dx_gen.shape[0]))

			# Train Generator
			x_gen = sample_noise(0.5, args.batch_size, args.noise_dim)
			discriminator_model.trainable = False
			gen_loss = gan_model.train_on_batch(x_gen, np.ones(x_gen.shape[0]))
			discriminator_model.trainable = True

			batch_counter += 1

			progress_bar.add(args.batch_size)

		print("{}/{} epochs completed".format(e, args.epochs))
		save_generated(generator_model, args.batch_size, args.noise_dim, e, data_set='mnist')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='A simple GAN')
	parser.add_argument('--epochs', type=int, default=400, help='The number of epochs to train for')
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--batch_per_epoch', type=int, default=200)
	parser.add_argument('--noise_dim', type=int, default=100)

	main(parser.parse_args())
