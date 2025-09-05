import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from AC_GAN import Discriminator, Generator

classifier = keras.models.load_model(r'../model/mnist_classifier_0.998.h5',
                                     custom_objects={'leaky_relu': tf.nn.leaky_relu})
CC = keras.losses.CategoricalCrossentropy(from_logits=False)


def get_special_category(dataset, category_index):
    x, y = dataset
    index = np.where(y == category_index)[0]
    return x[index]


def creat_noise(shape):
    # return np.random.uniform(-1, 1, shape)
    return np.random.normal(0, 1, shape)


def celoss_ones(logits):
    # loss = tf.losses.binary_crossentropy(logits, tf.ones_like(logits))
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.ones_like(logits) + np.random.normal(0, 0.1, logits.shape))

    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # loss = tf.losses.binary_crossentropy(logits, tf.zeros_like(logits))
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.zeros_like(logits) + np.random.normal(0, 0.1,
                                                                                                   logits.shape))
    return tf.reduce_mean(loss)


def g_loss_fn(discriminator, generator, img_num=1):
    fake_img = generator(creat_noise([img_num, 28, 28, 1]))
    class_loss = CC(classifier(fake_img), discriminator(fake_img)[1])
    logits = discriminator(fake_img)[0]
    loss = celoss_ones(logits)
    return (loss + class_loss) / 2


def d_loss_fn(discriminator, generator, real_img, img_num=1):
    fake_img = generator(creat_noise([img_num, 28, 28, 1]))
    d_real_logits = discriminator(real_img)[0]
    d_fake_logits = discriminator(fake_img)[0]
    class_loss = np.exp(-CC(classifier(fake_img), discriminator(fake_img)))
    fake_loss = celoss_zeros(d_fake_logits)
    real_loss = celoss_ones(d_real_logits)
    return (fake_loss + real_loss + class_loss) / 3


if __name__ == '__main__':
    batch_size = 1500
    epoch = 100
    iteration = 40
    (train_x, train_y), (_, _) = keras.datasets.mnist.load_data()
    generator = Generator().model()
    generator.summary()
    discriminator = Discriminator().model()
    discriminator.summary()
    for i in range(epoch):
        print('epoch=' + str(i))
        for j in range(iteration):
            with tf.GradientTape() as g:
                start = j * batch_size
                end = start + batch_size
                img_x = train_x[start:end]
                img_y = train_y[start:end]



