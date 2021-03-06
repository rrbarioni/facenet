"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            time_elapsed_log = open('validateLfw_timeElapsed_Log.txt', 'w')
            start_total_time = time.time()

            # Read the file containing the pairs used for testing
            print('Reading pairs from LFW')
            start_current_time = time.time()
            pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
            end_current_time = time.time()
            time_elapsed_log.write('Read pairs from LFW took ' + str(end_current_time - start_current_time)[0:5] + ' seconds\n')
            
            # Get the paths for the corresponding images
            print('Getting the paths for the corresponding images')
            start_current_time = time.time()
            paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)
            end_current_time = time.time()
            time_elapsed_log.write('Get the paths for the corresponding images took ' + str(end_current_time - start_current_time)[0:5] + ' seconds\n')
            
            # Load the model
            print('Loading model')
            start_current_time = time.time()
            facenet.load_model(args.model)
            end_current_time = time.time()
            time_elapsed_log.write('Load model took ' + str(end_current_time - start_current_time)[0:5] + ' seconds\n')
            
            # Get input and output tensors
            print('Getting input and output tensors')

            print('    images_placeholder')
            start_current_time = time.time()
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            end_current_time = time.time()
            time_elapsed_log.write('Get images placeholder took ' + str(end_current_time - start_current_time)[0:5] + ' seconds\n')
            
            print('    embeddings')
            start_current_time = time.time()
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            end_current_time = time.time()
            time_elapsed_log.write('Get embeddings took ' + str(end_current_time - start_current_time)[0:5] + ' seconds\n')
            
            print('    phase_train_placeholder')
            start_current_time = time.time()
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            end_current_time = time.time()
            time_elapsed_log.write('Get phase train placeholder took ' + str(end_current_time - start_current_time)[0:5] + ' seconds\n')
            
            print('Getting embedding_size')
            #image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
            start_current_time = time.time()
            image_size = args.image_size
            embedding_size = embeddings.get_shape()[1]
            end_current_time = time.time()
            time_elapsed_log.write('Get embedding size took ' + str(end_current_time - start_current_time)[0:5] + ' seconds\n')
        
            # Run forward pass to calculate embeddings
            print('Runnning forward pass on LFW images')
            start_forward_pass_time = time.time()
            batch_size = args.lfw_batch_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches):
                print('Batch ' + str(i+1) + '/' + str(nrof_batches))
                start_batch_time = time.time()
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]

                print('    Loading data')
                start_batch_loadData_time = time.time()
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                end_batch_loadData_time = time.time()
                time_elapsed_log.write('        Load data from batch ' + str(i+1) + '/' + str(nrof_batches) + ' took ' + str(end_batch_loadData_time - start_batch_loadData_time)[0:5] + ' seconds\n')

                print('    Running embeddings')
                start_batch_runEmbeddings_time = time.time()
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                end_batch_runEmbeddings_time = time.time()
                time_elapsed_log.write('        Run embeddings from batch ' + str(i+1) + '/' + str(nrof_batches) + ' took ' + str(end_batch_runEmbeddings_time - start_batch_runEmbeddings_time)[0:5] + ' seconds\n')
                end_batch_time = time.time()
                time_elapsed_log.write('    Batch ' + str(i+1) + '/' + str(nrof_batches) + ' took ' + str(end_batch_time - start_batch_time)[0:5] + ' seconds\n')
            end_forward_pass_time = time.time()
            time_elapsed_log.write('Run forward pass on LFW images took ' + str(end_forward_pass_time - start_forward_pass_time)[0:5] + ' seconds\n')
        
            start_evaluation_time = time.time()
            tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array, 
                actual_issame, nrof_folds=args.lfw_nrof_folds)
            end_evaluation_time = time.time()
            time_elapsed_log.write('Evaluation took ' + str(end_evaluation_time - start_evaluation_time)[0:5] + ' seconds\n')

            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

            auc = metrics.auc(fpr, tpr)
            print('Area Under Curve (AUC): %1.3f' % auc)
            eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
            print('Equal Error Rate (EER): %1.3f' % eer)

            end_total_time = time.time()
            time_elapsed_log.write('Total time was ' + str(end_total_time - start_total_time)[0:5] + ' seconds\n')
            time_elapsed_log.close()
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('lfw_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
