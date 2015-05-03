#!/usr/bin/python2

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
#%matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '../caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import os

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
IMAGE_FILES = sorted(map(lambda f: 'images/' + f, os.listdir('images')))
SYNSET_FILE = caffe_root + 'data/ilsvrc12/synset_words.txt'

classes = list(map(lambda l: ' '.join(l.split()[1:]), open(SYNSET_FILE)))

caffe.set_mode_cpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

input_images = map(lambda img_path: caffe.io.load_image(img_path), IMAGE_FILES)
#plt.imshow(input_images)

predictions = net.predict(input_images)  # predict takes any number of images, and formats them for the Caffe net automatically

for prediction in predictions:
	#plt.plot(prediction)
	print('predicted class: %s' % classes[prediction.argmax()])
	print('predicted probability: %.3f' % max(prediction))
	print('predicted entropy: %.3f' % entropy(prediction))
	print('')

