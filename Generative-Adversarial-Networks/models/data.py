import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import glob
from PIL import Image
import os


def normalization(x):
	"""
	Normalization for images
	:param x: The samples
	:return: The normalized samples
	"""
	x = x / 255.
	x = (x - 0.5) / 0.5
	return x


def inverse_normalization(x):
	"""
	Inverse operation of the normalization function
	:param x: The normalized samples
	:return: Un-normalized samples
	"""
	x = x * 0.5 + 0.5
	x = x * 255.
	return x.astype(np.uint8)


def load_data(data_set_name):
	"""
	Load some data sets
	:param data_set_name: The data set we want
	:return: The loaded examples
	"""
	if data_set_name == 'mnist':
		(X_train, _), (X_test, _) = mnist.load_data()
		x = np.concatenate((X_train, X_test))
		x = normalization(x)
		x = np.resize(x, (x.shape[0], x.shape[1], x.shape[2], 1))

		padded = np.zeros((x.shape[0], 64, 64, 1))

		for sample_id in range(x.shape[0]):
			for row in range(x.shape[1]):
				for col in range(x.shape[2]):
					padded[sample_id, row, col, :] = x[sample_id, row, col, :]

		plt.imshow(np.resize(padded[0], (64, 64)), cmap='gray')
		plt.savefig('Generative-Adversarial-Networks/generated_images/mnist_test.png')
		plt.clf()
		plt.close()

		return padded
	elif data_set_name == 'pokemon':
		filelist = glob.glob('Generative-Adversarial-Networks/data/pokemon-gen1/*.png')
		x = np.array([np.array(Image.open(fname)) for fname in filelist])

		plt.imshow(x[0])
		plt.savefig('Generative-Adversarial-Networks/generated_images/pokemon_test.png')
		plt.clf()
		plt.close()

		x = normalization(x)

		return x
	elif data_set_name == 'pokemon5':
		filelist = glob.glob('Generative-Adversarial-Networks/data/pokemon-gen5-sized/*.png')
		x = np.array([np.array(Image.open(fname).convert('RGB')) for fname in filelist])

		plt.imshow(x[0])
		plt.savefig('Generative-Adversarial-Networks/generated_images/pokemon5_test.png')
		plt.clf()
		plt.close()

		x = normalization(x)

		return x
	else:
		ValueError('Not a valid data set')


def generate_batch(x, batch_size):
	"""
	Generator function for getting batches
	:param x: The full collection of samples
	:param batch_size: The batch size desired
	:return: A generated batch
	"""
	while True:
		i = np.random.choice(x.shape[0], batch_size, replace=False)
		yield x[i]


def sample_noise(noise_scale, batch_size, noise_dim):
	"""
	Function for getting random noise
	:param noise_scale: The variance of noise
	:param batch_size: Number of noise vectors
	:param noise_dim: Size of noise vector
	:return: Batch of noise vectors
	"""
	return np.random.normal(scale=noise_scale, size=(batch_size, noise_dim))


def get_d_batch(x_real, generator_model, batch_size, noise_dim, noise_scale=0.5):
	"""
	Gets a batch of samples for the discriminator to train on
	:param x_real: The collection of real samples
	:param generator_model: The generator portion of the model
	:param batch_size: The size of the batch desired
	:param noise_dim: The dimension of the noise vector
	:param noise_scale: The variance of the noise
	:return: A tuple of real and fake samples
	"""
	noise_input = sample_noise(noise_scale, batch_size, noise_dim)
	dx_fake = generator_model.predict(noise_input, batch_size=batch_size)
	dx_real = x_real[:batch_size]

	return dx_real, dx_fake


def save_generated(generator_model, batch_size, noise_dim, epoch, noise_scale=0.5, image_output=(64, 64), data_set='mnist', model_name='gan'):
	"""
	Saves a collection of generated images as 1 big image
	:param generator_model: The generator model to generate images with
	:param batch_size: The batch size
	:param noise_dim: The dimension of the noise vector
	:param epoch: Current epoch of training
	:param noise_scale: The variance of the noise
	:param image_output: The image output size
	:param data_set: The data set that's being worked with
	:param model_name: The name of the particular model
	:return: Nothing
	"""
	noise_input = sample_noise(noise_scale, batch_size, noise_dim)
	x_gen = generator_model.predict(noise_input)
	x_gen = inverse_normalization(x_gen)

	fig = plt.figure(figsize=(8, 8))
	columns = 4
	rows = 4
	x_to_save = x_gen[:columns * rows]
	for i in range(1, columns * rows + 1):
		fig.add_subplot(rows, columns, i)
		x = np.resize(x_to_save[i-1], image_output)
		if data_set == 'mnist':
			x = np.resize(x_to_save[i - 1], (64, 64))
			plt.imshow(x, cmap='gray')
		else:
			plt.imshow(x)
		plt.axis('off')
	plt.savefig('Generative-Adversarial-Networks/generated_images/{}_{}_-epoch-{}.png'.format(model_name, data_set, epoch))
	plt.clf()
	plt.close()


def save_model_weights(generator_model, discriminator_model, gan_model, e, model_name='gan', data_set='mnist'):
	"""
	Saves the current weights of the model
	:param generator_model: The generator portion of the model
	:param discriminator_model: The discriminator portion of the model
	:param gan_model: The full gan
	:param e: The current epoch
	:param model_name: The name of the model we are working with
	:param data_set: The data set we are working with
	:return: Nothing
	"""
	model_path = '/home/hunter/git/ML-Open-Source-Implementations/Generative-Adversarial-Networks/models/weights/'

	if e % 5 == 0:
		gen_weights_path = os.path.join(model_path, '{}-gen_epoch{}_{}.h5'.format(model_name, e, data_set))
		generator_model.save_weights(gen_weights_path, overwrite=True)

		disc_weights_path = os.path.join(model_path, '{}-disc_epoch{}_{}.h5'.format(model_name, e, data_set))
		discriminator_model.save_weights(disc_weights_path, overwrite=True)

		gan_weights_path = os.path.join(model_path, '{}-full_epoch{}_{}.h5'.format(model_name, e, data_set))
		gan_model.save_weights(gan_weights_path, overwrite=True)
