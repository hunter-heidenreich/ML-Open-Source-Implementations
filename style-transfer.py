# Import VGG19
###
# Content loss -> L_cont(original, generated, layer) = 1/2 * SUM(GEN - ORIG)^2, where caps are feature reps at layer
# Content loss derivative -> For each i,j if GEN[i,j] < 0, = 0 ... else (GEN[i,j] - ORIG[i,j]
###
# Gram matrix -> G[i,j]^l = SUM[k](GEN[i,k]^l * GEN[j,k]^l)
# Entries = E_l = (1/(4 * N_l^2 * M_l^2)) * (SUM[i,j] (GEN[i,j] - ORIG[i,j])^2)
# Style loss -> SUM_(l=0)^L w_l * E_l
# Style loss derivative -> 0 if F[i,j]_l < 0, else (1/(N_l^2 * M_l^2)) * ((F_l)^T * (GEN_l - ORiG_l))
###
# Total loss -> L(CONT, STYL, GEN) = a * L_cont(CONT, GEN) + b * L_style(STYL, GEN)

import keras.backend as K
from keras.applications import VGG16

from PIL import Image

import numpy as np
import time

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave


# Params #

img_height = 512
img_width = 512
img_size = img_height * img_width
img_channels = 3

content_path = 'images/output_3_0.png'
style_path = 'images/output_4_0.png'
target_path = 'generated-image'
target_extension = '.png'

CONTENT_IMAGE_POS = 0
STYLE_IMAGE_POS = 1
GENERATED_IMAGE_POS = 2

# Params #


def process_img(path):
    """
    Function for processing images to the format we need
    :param path: The path to the image
    :return: The image as a data array, scaled and reflected
    """
    # Open image and resize it
    img = Image.open(path)
    img = img.resize((img_width, img_height))

    # Convert image to data array
    data = np.asarray(img, dtype='float32')
    data = np.expand_dims(data, axis=0)

    # Apply pre-process to match VGG16 we are using
    data[:, :, :, 0] -= 103.939
    data[:, :, :, 1] -= 116.779
    data[:, :, :, 2] -= 123.68

    # Flip from RGB to BGR
    data = data[:, :, :, ::-1]

    return data


def get_layers(content_matrix, style_matrix, generated_matrix):
    # Prep the model for our new input sizes
    input_tensor = K.concatenate([content_matrix, style_matrix, generated_matrix], axis=0)
    model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # Convert layers to dictionary
    layers = dict([(layer.name, layer.output) for layer in model.layers])

    # Pull the specific layers we want
    c_layers = layers['block2_conv2']
    s_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
    s_layers = [layers[layer] for layer in s_layers]

    return c_layers, s_layers


def content_loss(content_features, generated_features):
    return 0.5 * K.sum(K.square(generated_features - content_features))


def gram_matrix(features):
    """
    Calculates the gram matrix of the feature representation matrix
    :param features: The feature matrix that is used to calculate the gram matrix
    :return: The gram matrix
    """
    return K.dot(features, K.transpose(features))


def style_loss(style_matrix, generated_matrix):
    """
    Computes the style loss of the transfer
    :param style_matrix: The style representation from the target style image
    :param generated_matrix: The style representation from the generated image
    :return: The loss from the style content
    """
    # Permute the matrix to calculate proper covariance
    style_features = K.batch_flatten(K.permute_dimensions(style_matrix, (2, 0, 1)))
    generated_features = K.batch_flatten(K.permute_dimensions(generated_matrix, (2, 0, 1)))

    # Get the gram matrices
    style_mat = gram_matrix(style_features)
    generated_mat = gram_matrix(generated_features)

    return K.sum(K.square(style_mat - generated_mat)) / (4.0 * (img_channels ** 2) * (img_size ** 2))


def total_loss(c_layer, s_layers, alpha=1.0, beta=10000.0):
    # Content loss
    content_features = c_layer[CONTENT_IMAGE_POS, :, :, :]
    generated_features = c_layer[GENERATED_IMAGE_POS, :, :, :]
    c_loss = content_loss(content_features, generated_features)

    # Style loss
    s_loss = None
    for layer in s_layers:
        style_features = layer[STYLE_IMAGE_POS, :, :, :]
        generated_features = layer[GENERATED_IMAGE_POS, :, :, :]
        if s_loss is None:
            s_loss = style_loss(style_features, generated_features)
        else:
            s_loss += style_loss(style_features, generated_features)

    return alpha * c_loss + (beta / len(s_layers)) * s_loss


def eval_loss_and_grads(generated):
    generated = generated.reshape((1, img_height, img_width, 3))
    outs = f_outputs([generated])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values


def save_image(filename, generated):
    # Reshape image and flip from BGR to RGB
    generated = generated.reshape((img_height, img_width, 3))
    generated = generated[:, :, ::-1]

    # Re-apply the mean shift
    generated[:, :, 0] += 103.939
    generated[:, :, 1] += 116.779
    generated[:, :, 2] += 123.68

    # Clip values to 0-255
    generated = np.clip(generated, 0, 255).astype('uint8')

    imsave(filename, Image.fromarray(generated))


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grad_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


if __name__ == '__main__':
    # Prepare the generated image
    generated_img = np.random.uniform(0, 255, (1, img_height, img_width, 3)) - 128.

    name = '{}-{}{}'.format(target_path, 0, target_extension)
    save_image(name, generated_img)

    # Load the respective content and style images
    content = process_img(content_path)
    style = process_img(style_path)

    # Prepare the variables for the flow graph
    content_image = K.variable(content)
    style_image = K.variable(style)
    generated_image = K.placeholder((1, img_height, img_width, 3))
    loss = K.variable(0.)

    # Grab the layers needed to prepare the loss metric
    content_layer, style_layers = get_layers(content_image, style_image, generated_image)

    # Define loss and gradient
    loss = total_loss(content_layer, style_layers)
    grads = K.gradients(loss, generated_image)

    # Define the output
    outputs = [loss]
    outputs += grads
    f_outputs = K.function([generated_image], outputs)

    evaluator = Evaluator()
    iterations = 10

    for i in range(iterations):
        print('Start of iteration', i)
        start_time = time.time()
        generated_img, min_val, info = fmin_l_bfgs_b(evaluator.loss, generated_img.flatten(),
                                                     fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        end_time = time.time()
        print('Iteration {} completed in {}'.format(i, end_time - start_time))
        name = '{}-{}{}'.format(target_path, i+1, target_extension)
        save_image(name, generated_img)
        print('Saved image to: {}'.format(name))