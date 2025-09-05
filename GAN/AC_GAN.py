import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = layers.Conv2D(16, kernel_size=3, strides=1, input_shape=(28, 28, 1), padding="same")
        self.drop = layers.Dropout(0.25)

        self.conv2 = layers.Conv2D(32, kernel_size=3, strides=1, padding="same")
        self.drop1 = layers.Dropout(0.25)
        self.bn = layers.BatchNormalization(momentum=0.8)

        self.conv3 = layers.Conv2D(64, kernel_size=3, strides=1, padding="same")
        self.drop2 = layers.Dropout(0.25)
        self.bn2 = layers.BatchNormalization(momentum=0.8)

        self.conv4 = layers.Conv2D(128, kernel_size=3, strides=1, padding="same")
        self.drop3 = layers.Dropout(0.25)
        self.bn3 = layers.BatchNormalization(momentum=0.8)

        self.conv5 = layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.drop4 = layers.Dropout(0.25)
        self.bn4 = layers.BatchNormalization(momentum=0.8)

        self.conv6 = layers.Conv2D(512, kernel_size=3, strides=1, padding="same")
        self.drop5 = layers.Dropout(0.25)

        self.gap = layers.GlobalAveragePooling2D()
        self.fc2 = layers.Dense(1)
        self.fc3 = layers.Dense(10)

    def call(self, inputs, training=None, mask=None):
        x = self.drop(tf.nn.leaky_relu(self.conv1(inputs), alpha=0.2))
        x = self.bn(self.drop1(tf.nn.leaky_relu(self.conv2(x), alpha=0.2)))
        x = self.bn2(self.drop2(tf.nn.leaky_relu(self.conv3(x), alpha=0.2)))
        x = self.drop3(tf.nn.leaky_relu(self.conv4(x), alpha=0.2))
        x = self.gap(x)
        valide = tf.nn.sigmoid(self.fc2(x))
        category = tf.nn.softmax(self.fc3(x))
        return valide, category

    def model(self):
        x = layers.Input(shape=(28, 28, 1))
        return keras.Model(inputs=x, outputs=self.call(x))


class Generator(keras.Model):

    def __init__(self):
        super(Generator, self).__init__()
        self.conv = layers.Conv2D(32, kernel_size=5, input_shape=(28, 28, 1))
        self.bn = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(64, kernel_size=5)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(128, kernel_size=5)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(64, kernel_size=3)
        self.bn3 = layers.BatchNormalization()
        self.up_sample3 = layers.UpSampling2D()
        self.conv4 = layers.Conv2D(32, kernel_size=3, padding="same")
        self.bn4 = layers.BatchNormalization()
        self.conv5 = layers.Conv2D(1, kernel_size=3, padding="same")

    def call(self, inputs, training=None, mask=None):
        x = self.bn(self.conv(inputs))
        x = self.bn1(tf.nn.leaky_relu(self.conv1(x)))
        x = self.bn2(tf.nn.leaky_relu(self.conv2(x)))
        x = self.up_sample3(self.bn3(tf.nn.leaky_relu(self.conv3(x))))
        x = self.bn4(tf.nn.leaky_relu(self.conv4(x)))
        x = tf.nn.tanh(self.conv5(x))
        return x

    def model(self):
        x = layers.Input(shape=(28, 28, 1))
        return keras.Model(inputs=x, outputs=self.call(x))
    # def get_combine(self):
    #     noise = layers.Input(shape=(100,))
    #     label = layers.Input(shape=(1,), dwtype='int32')
    #     fake_img = self.call([noise, label])
    #     discriminator = Discriminator()
    #     discriminator.trainable = False
    #     valide, lable = discriminator(fake_img)
    #     return keras.Model(inputs=[noise, label], outputs=[valide, lable])

# generator = Generator().model()
# # generator.summary()
# y = generator([np.random.normal(0, 1, [1, 100])])
# y = y[0].numpy()
# y = (y + 1) * 127.5
# y = y.astype(int)
# plt.imshow(y.reshape(28, 28), cmap='gray')
# plt.show()
