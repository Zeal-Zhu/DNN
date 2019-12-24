import pylab
import scipy.misc
import time
import numpy as np
import h5py
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_methods import *

#%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#%load_ext autoreload
#%autoreload 2

np.random.seed(1)

train_dataset_file = 'datasets/train_catvnoncat.h5'
test_dataset_file = 'datasets/test_catvnoncat.h5'

train_x_orig, train_y, test_x_orig, test_y, classes = load_data(
    train_dataset_file, test_dataset_file)

# Example of a picture
index = 2
#plt.imshow(train_x_orig[index])
print("y = " + str(train_y[0, index]) + ". It's a " +
      classes[train_y[0, index]].decode("utf-8") + " picture.")

# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print("Number of training examples: " + str(m_train))
print("Number of testing examples: " + str(m_test))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_x_orig shape: " + str(train_x_orig.shape))
print("train_y shape: " + str(train_y.shape))
print("test_x_orig shape: " + str(test_x_orig.shape))
print("test_y shape: " + str(test_y.shape))

# Reshape the training and test examples
# The "-1" makes reshape flatten the remaining dimensions
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))

### CONSTANTS ###
layers_dims = [12288, 20, 20,30, 7, 5, 1]  # 6-layer model

print("# of layers: "+str(len(layers_dims)))


matplotlib.rcParams.update({'font.size': 5}) 

parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate=0.000001, num_iterations=2000, print_cost=True)

pred_train = predict(train_x, train_y, parameters)

pred_test = predict(test_x, test_y, parameters)

num_mis_img = print_mislabeled_images(classes, test_x, test_y, pred_test)

# print("missing rate = "+(num_mis_img/int(m_test)))

## START CODE HERE ##
my_image = "my_image1.jpg"  # change this to the name of your image file
my_label_y = [1]  # the true class of your image (1 -> cat, 0 -> non-cat)
## END CODE HERE ##

fname = "images/" + my_image
image = np.array(plt.imread(fname))
# my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((num_px*num_px*3, 1))
my_image=np.array(Image.fromarray(image).resize(size=(num_px, num_px))).reshape((num_px*num_px*3, 1))
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
information = "y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)), ].decode("utf-8") + "\" picture."
#plt.title(fname + "\n" + information)
print(information)
pylab.show()
