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

def costs(class_ind, prediction):
	res = 0

	i = 0
	for prob in prediction:
		if i == class_ind:
			res += (prob - 1)**2
		else:
			res += prob**2
	
	return res

if __name__ == '__main__':
	classes = list(map(lambda l: ' '.join(l.split()[1:]), open(SYNSET_FILE)))

	caffe.set_mode_cpu()
	net = caffe.Classifier(MODEL_FILE, PRETRAINED,
			       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
			       channel_swap=(2,1,0),
			       raw_scale=255,
			       image_dims=(256, 256))

	print('==> loading images...')
	input_images = map(lambda img_path: caffe.io.load_image(img_path), IMAGE_FILES)
	input_classes = [336, None, None, None, None, 948, 458, None]
	#plt.imshow(input_images)

	print('==> finished loading')
	print('==> analyzing images')
	predictions = net.predict(input_images)  # predict takes any number of images, and formats them for the Caffe net automatically

	# output
	if len(sys.argv) < 2:
		# screen output
		i = 0
		for prediction in predictions:
			print('')
			print('image: %s' % IMAGE_FILES[i].split('/')[-1])
			print('\tclass: %s' % classes[prediction.argmax()])
			print('\tprobability: %.3f' % max(prediction))
			print('\tentropy: %.3f' % entropy(prediction))
			i += 1
	elif sys.argv[1] == 'csv':
		with open('results.csv', 'w') as f:
			f.write('file name,class,probability,entropy\n')
			# i: image counter
			i = 0
			for prediction in predictions:
				#plt.plot(prediction)
				f.write('\\texttt{%s},' % (IMAGE_FILES[i].split('/')[-1].replace('_', '\\_'),))
				f.write(classes[prediction.argmax()].split(',')[0] + ',')
				f.write('%.3f,' % max(prediction))
				f.write('%.3f\n' % entropy(prediction))
				#if input_classes[i] is not None:
					#print('\tclassification costs: %.3f' % costs(input_classes[i]-1, prediction))
				i += 1
	elif sys.argv[1] == 'latex':
		with open('results.tex', 'w') as f:
			f.write('\\begin{tabular}{clll}\n')
			f.write('file name&class&probability&entropy \\\\\n')
			f.write('\\hline')
			i = 0
			for prediction in predictions:
				f.write('\\texttt{%s}&' % (IMAGE_FILES[i].split('/')[-1].replace('_', '\\_'),))
				f.write(classes[prediction.argmax()] + '&')
				f.write('%.3f&' % max(prediction))
				f.write('%.3f \\\\\n' % entropy(prediction))
				i += 1
			f.write('\\end{tabular}\n')

