'''This script shows how to build a Deep AutoEncoder  
to reconstruct MNIST digit dataset using Keras library
'''
from keras.layers import Input, Dense
from keras.models import Model

#this is a size of encoded representation
encoding_dim = 32
image_dim = 784

#this is our input placeholder
input_img = Input(shape=(image_dim,))
#encoded is a encoded representation of the input
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
#decoded is the lossy reconstruction of the input
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
deep_autoencoder = Model(input_img, decoded)
#this model maps an input to  its encoded representation - outputs 32 dmin
encoder = Model(input_img, encoded)

#create a placeholder for a encoded (32-dimension) input
encoded_input = Input(shape=(32,))
#retrieve the last layer of the deep_autoencoder model - output 64 dim, after 128, after 784
decoder_layer1 = deep_autoencoder.layers[-3]
decoder_layer2 = deep_autoencoder.layers[-2]
decoder_layer3 = deep_autoencoder.layers[-1]

decoder_layer = decoder_layer3(decoder_layer2(decoder_layer1(encoded_input)))

#create the  decoder model
decoder = Model(encoded_input, decoder_layer)

#train the deep_autoencoder
deep_autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#######################################################################
# preparing dataset. discarding labels
from keras.datasets import mnist
import numpy as np 
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test),np.prod(x_train.shape[1:])))

print(x_train.shape)
print(x_test.shape)

deep_autoencoder.fit(x_train,x_train,epochs=100,
	batch_size=256,
	shuffle=True,
	validation_data=(x_test,x_test))

# encoded and decoded some digits
# note that we take them from the test dataset

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

#use matplotlib
import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
	#display original
	ax = plt.subplot(2,n,i+1)
	plt.imshow(x_test[i].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	#display reconstruction
	ax = plt.subplot(2,n,i+1+n)
	plt.imshow(decoded_imgs[i].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

plt.show()
