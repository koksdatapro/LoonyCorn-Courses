import os
import numpy as np

def load_dataset():
	def download(filename, source = 'http://yann.lecun.com/exdb/mnist/'):
		print ("Downloading ", filename)
		import urllib
		urllib.urlretrieve(source+filename,filename)


	import gzip
		
	def load_mnist_images(filename):
		if not os.path.exists(filename):
			download(filename)


		with gzip.open(filename, 'rb') as f:

			data=np.frombuffer(f.read(), np.uint8, offset=16)

			data = data.reshape(-1,1,28,28)

			return data/np.float32(256)

	def load_mnist_labels(filename):

		if not os.path.exists(filename):

			download(filename)

		with gzip.open(filename, 'rb') as f:

			data = np.frombuffer(f.read(), np.uint8, offset = 8)

		return data


	X_train = load_mnist_images('train-images-idx3-ubyte.gz')
	y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
	X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
	y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
	    
	return X_train, y_train, X_test,  y_test




X_train, y_train, X_test, y_test = load_dataset()
