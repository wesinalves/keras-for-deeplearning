'''This script shows how to build a Convolutional AutoEncoder  
to reconstruct MNIST digit dataset using Keras library
'''
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

input_img = Input(shape=(28,28,1))

x = Conv2D(16,(3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8,(3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8,(3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)

# at this point the represatation is (4, 4, 8), it is 128 dimensional

x = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(16, (3,3), activation='relu')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

conv_autoencoder = Model(input_img, decoded)
conv_autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = np.reshape(x_train, (len(x_train),28,28,1)) #adapt this if using 'channel_first' image data format
x_test = np.reshape(x_test, (len(x_test),28,28,1)) #adapt this if using 'channel_first' image data format

conv_autoencoder.fit(x_train, x_train,
	epochs=3,
	batch_size=128,
	shuffle=True,
	validation_data=(x_test,x_test))

import matplotlib.pyplot as plt

decoded_imgs = conv_autoencoder.predict(x_test)
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
	#display original image
	ax = plt.subplot(2,n,i+1)
	plt.imshow(x_test[i].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	#display reconstruct image
	ax = plt.subplot(2, n, i+n+1)
	plt.imshow(decoded_imgs[i].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

plt.show()

