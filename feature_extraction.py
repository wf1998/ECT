import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from nets_VAE import MY, model_concat, Decoder, Encoder,Classifier
from nets_CVAE import BIGVAE
from tensorflow.keras import backend as K

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def get_special_category(dataset, category_index):
    x, y = dataset
    index = np.where(y == category_index)[0]
    return x[index]


def showimg(img):
    plt.imshow(np.asarray(img).reshape(28, 28))
    plt.show()


if __name__ == '__main__':
    # 先训练分类器
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    train_x = train_x / 255.0
    test_x = test_x / 255.0
    train_x = train_x.reshape(-1, 28, 28, 1)
    test_x = test_x.reshape(-1, 28, 28, 1)
    special_category = get_special_category((train_x, train_y), 0)
    classifier_optimizer = keras.optimizers.Adam(lr=0.02)
    classifier = Classifier().model()
    classifier.summary()
    classifier.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=classifier_optimizer,
                       metrics=['accuracy'])
    callback = keras.callbacks.ModelCheckpoint(filepath='F:/搜狗下载/vae-master/my_vae/model/classifier_{accuracy:.3f}.h5',
                                               save_best_only=True,
                                               mode='max',
                                               monitor='val_accuracy', verbose=0)
    classifier.fit(epochs=1, x=train_x, y=train_y, validation_data=(test_x, test_y))
    classifier = keras.models.load_model('model/classifier_0.922.h5',
                                         custom_objects={"leaky_relu": tf.nn.leaky_relu, 'tanh': tf.nn.tanh})
    # 训练变分自编码器
    vae_optimizer = keras.optimizers.RMSprop(learning_rate=0.002)
    vae = MY().model()
    vae._name = 'vae'
    classifier._name = 'classifier'
    classifier.summary()
    vae.summary()
    classifier.trainable = False
    x = keras.layers.Input(shape=(28, 28, 1))
    x_ = vae(x)
    feature = vae.output
    y = classifier(x_)
    combined = keras.Model(inputs=x, outputs=[y, feature])
    combined.summary()
    combined.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=vae_optimizer,
                     metrics=['accuracy'])
    callback = keras.callbacks.ModelCheckpoint(filepath='F:/搜狗下载/vae-master/my_vae/model/combined_{accuracy:.3f}.h5',
                                               save_best_only=True,
                                               mode='max',
                                               monitor='val_accuracy', verbose=0)
    combined.fit(epochs=10, x=train_x, y=train_y, callbacks=callback, validation_data=(test_x, test_y))

    SCC = keras.losses.SparseCategoricalCrossentropy()
    # img_y = tf.one_hot(np.zeros(shape=50), 10)
    for i in range(10000):
        with tf.GradientTape() as tape:
            random_index = np.random.randint(low=0, high=60000, size=50)
            img_x = train_x[random_index]
            img_y = train_y[random_index]
            y_ = combined(img_x)
            # img_y = classifier(img_x)
            lable_loss = SCC(y_, img_y)
            # lable_loss = tf.reduce_mean((y_ - img_y) ** 2)
            # vision_loss = tf.reduce_mean((img_x - x_) ** 2)
            loss = lable_loss

            grads = tape.gradient(loss, vae.trainable_variables)
            vae_optimizer.apply_gradients(zip(grads, vae.trainable_variables))
            # print('\t loss=%.5f,lable_loss=%.5f,vision_loss=%.5f' % (loss, lable_loss, vision_loss))
            print('\t loss=%.30f,lable_loss=%.30f' % (loss, lable_loss))
