from sklearn.model_selection import train_test_split
import numpy as np
import time
import argparse
from datetime import datetime
import h5py
import tensorflow as tf
import pickle
from dataPreprocessor import Processor, DataGenerator_G, DataGenerator_D
from util_generator import set, encoder, decoder, fuse3D
import os
from tqdm import tqdm
from dataPreprocessor import Processor

### Set Parameters
opt = set()
opt.batchSize = 1
LOGDIR = './save'

### Build Graph
tf.compat.v1.reset_default_graph()
g= tf.Graph()
with g.as_default():
    inputImage = tf.compat.v1.placeholder(tf.float32, shape=[opt.batchSize, opt.inH, opt.inW, 3])

### Build Generator Module
def build_generator(inputImage):
    with tf.compat.v1.variable_scope('generator'):
        for i in range(3):
            singleImage = inputImage[:, :, :, i:i+1] #[B,192,256,1]
            latent = encoder(opt, singleImage, i) #[B,512]
            XYZ = decoder(opt, latent, i)  # [B,H,W,3V] (B, 15, 20, 24)
            fuseTrans = tf.nn.l2_normalize(opt.fuseTrans, axis=1)
            if i == 0 :
                XYZid_0 = fuse3D(opt, XYZ, fuseTrans) #[B,3,2400]
            elif i == 1 :
                XYZid_1 = fuse3D(opt, XYZ, fuseTrans) #[B,3,2400]
            else:
                XYZid_2 = fuse3D(opt, XYZ, fuseTrans) #[B,3,2400]
        XYZid = tf.concat([XYZid_0, XYZid_1, XYZid_2], axis=2) #[B,3,N=7200]
        return XYZid, latent, XYZ, inputImage

### Load Weight and Run Generator Module
with g.as_default():
    generated_ptcloud, encoded, decoded, originalItems = build_generator(inputImage)
    sess = tf.compat.v1.InteractiveSession()
    latest_checkpoint = tf.compat.v1.train.latest_checkpoint(LOGDIR + "/checkpoints/")
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, latest_checkpoint)

if __name__=='__main__':
    ### Set Configurations below
    params = {'epochs': 1,           
            'batch_size': 1,         
            'n_classes': 2,
            'width': 256,
            'height': 192,
            'n_channels': 3,
            'label': False,           #是否需要y=0的label. If 'True' ,data generator output = [img, pointcloud, y], otherwise, [img, pointcloud]
            'shuffle': True,
            'N': 2400
            }
    path_image = './test_images/001851'   # 2D image data
    path_pointcloud = './output_test.xyz'      # 3D pointcloud data
    path_pointclouds = './testoutput/output1' + '.xyz' # output for pointcloud
    narrowdown = True
    REAL3D = False

    ### Load Testing Image
    item = Processor() 
    images = item.test(path_image,path_pointcloud, narrowdown, REAL3D, params['width'], 
                        params['n_channels'], params['N']) # pointcloud_real (192, 256, 3)
    images = images.reshape(1, 192, 256, 3)
    print(images.shape)
    print('Begin testing...')
    out_path = './testoutput'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    ### Generate Output
    [opt_ptcloud, opt_enc, opt_dec] = sess.run([generated_ptcloud, encoded, decoded],feed_dict={inputImage: images})
    print(opt_ptcloud.shape)
    small_ptcloud = opt_ptcloud[0][:, :]
    mat = np.matrix(small_ptcloud.T)
    with open(path_pointclouds, 'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')
    print('finished!...')
