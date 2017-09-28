"""Performs face alignment and stores face thumbnails in the output directory."""
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

import sys
sys.path.append('src/')

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import simplified_facenet as facenet
import align.detect_face
import random
from time import sleep

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(args):
    print('Creating networks and loading parameters')
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=False)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ] # three steps's threshold
    factor = 0.709 # scale factor
    
    input_file = args.input_file
    output_file = args.output_file
    img = misc.imread(input_file)
    img = img[:,:,0:3]

    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    # print(str(bounding_boxes))
    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces > 0:
        det = bounding_boxes[:,0:4]
        img_size = np.asarray(img.shape)[0:2]
        if nrof_faces>1:
            bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
            img_center = img_size / 2
            offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
            offset_dist_squared = np.sum(np.power(offsets,2.0),0)
            index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
            det = det[index,:]
        det = np.squeeze(det)
        det = [int(round(d)) for d in det]
        cropped = img[det[1]:det[3],det[0]:det[2],:]
        misc.imsave(output_file, cropped)
    else:
        print('Unable to align "%s"' % input_file)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_file', type=str, help='File with unaligned image.')
    parser.add_argument('output_file', type=str, help='File with aligned face thumbnails.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
