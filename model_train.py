from sklearn.model_selection import train_test_split
import numpy as np
import time
import argparse
from datetime import datetime
import h5py
import tensorflow as tf
import pickle
from dataPreprocessor import Processor, DataGenerator_G, DataGenerator_D
import util_discriminator as tf_util
from util_generator import set, encoder, decoder, fuse3D

opt = set()
batch_size =20
num_pointcloud = 2400
G_LEARNING_RATE = 0.0004
D_LEARNING_RATE = 0.00001
LOGDIR='./save'
RESTORE = True
TRAINING = True
epochs_list= [] # store the epoch corresponding to the variables below
gen_loss = []
dis_loss = []
true_acc = []
fake_acc = []
tot_acc  = []
ptcloud_list = []
# Build Graph
tf.compat.v1.reset_default_graph()
g= tf.Graph()
with g.as_default():
    inputImage = tf.compat.v1.placeholder(tf.float32, shape=[opt.batchSize, opt.inH, opt.inW, 3])
    inputPtcloud = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, 3, num_pointcloud])
    inputPtcloud_new = inputPtcloud + tf.random.normal(shape = tf.shape(inputPtcloud), mean = 0.0, stddev = 0.1, dtype = tf.float32)
def build_generator(inputImage):
    with tf.compat.v1.variable_scope('generator'):
        # XYZid = tf.Variable(np.empty((inputImage.shape[0],3,0), dtype=np.float32))
        # print(XYZid.shape)
        for i in range(3):
            singleImage = inputImage[:, :, :, i:i+1] #[B,192,256,1]
            #print(i, singleImage.shape)
            latent = encoder(opt, singleImage, i) #[B,512] 
            XYZ = decoder(opt, latent, i)  # [B,H,W,3V] (B, 15, 20, 24)
            fuseTrans = tf.nn.l2_normalize(opt.fuseTrans, axis=1)
            if i == 0 :
                XYZid_0 = fuse3D(opt, XYZ, fuseTrans)[:,:,0::3] #[B,3,800]
            elif i == 1 :
                XYZid_1 = fuse3D(opt, XYZ, fuseTrans)[:,:,1::3] #[B,3,800]
            else:
                XYZid_2 = fuse3D(opt, XYZ, fuseTrans)[:,:,2::3] #[B,3,800]
        XYZid = tf.concat([XYZid_0, XYZid_1, XYZid_2], axis=2) #[B,3,N=2400]
        # print('xyzid',XYZid.shape)
        return XYZid, latent, XYZ, inputImage

def build_discriminator(point_cloud,reuse=True):
    batch_size = 20
    num_point = 512
    bn_decay = None
    is_training = tf.cast(TRAINING,tf.bool)
    with tf.compat.v1.variable_scope('discriminator',reuse=reuse):
        input_image = tf.expand_dims(point_cloud, -1)

        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(input_image, 64, [1, 3],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)

        # Symmetric function: max pooling
        net = tf.compat.v1.transpose(net,perm = [0,2,3,1])

        net = tf_util.max_pool2d(net, [1, num_point],
                                 padding='VALID', scope='maxpool')

        # MLP on global point cloud vector
        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                      scope='fc1', bn_decay=bn_decay)
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                      scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
        net = tf_util.fully_connected(net, 1, activation_fn=None, scope='fc3')

        outnet = tf.nn.sigmoid(net, name="discriminator_sigmoid")
        return outnet*0.9



with g.as_default():
    real_output = build_discriminator(inputPtcloud_new,reuse=False) 
    generated_ptcloud, encoded, decoded, in_img = build_generator(inputImage)
    fake_output = build_discriminator(generated_ptcloud,reuse = True)#True)

    with tf.name_scope("cross_entropy") as scope:
        d_loss_total = -tf.reduce_mean(tf.math.log(tf.math.maximum(real_output,1e-9)) + tf.math.log(tf.math.maximum(1. - fake_output,1e-9)))
        g_loss = -tf.reduce_mean(tf.math.log(tf.math.maximum(fake_output,1e-9)))
        d_loss_summary = tf.compat.v1.summary.scalar("Discriminator_Total_Loss", d_loss_total)
        d_loss_summary = tf.compat.v1.summary.scalar("Generator_Loss", g_loss)

    with tf.name_scope("accuracy") as scope:
        # Compute the discriminator accuracy on real data, fake data, and total:
        accuracy_real = tf.reduce_mean(tf.cast(tf.equal(tf.round(real_output),
                                                        tf.ones_like(real_output)),
                                               tf.float32))
        accuracy_fake = tf.reduce_mean(tf.cast(tf.equal(tf.round(fake_output),
                                                        tf.zeros_like(fake_output)),
                                               tf.float32))

        total_accuracy = 0.5 * (accuracy_fake + accuracy_real)
        acc_real_summary = tf.compat.v1.summary.scalar("Real_Accuracy", accuracy_real)
        acc_real_summary = tf.compat.v1.summary.scalar("Fake_Accuracy", accuracy_fake)
        acc_real_summary = tf.compat.v1.summary.scalar("Total_Accuracy", total_accuracy)
    with tf.name_scope("training") as scope:
        # Global steps are useful for restoring training:
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        # Make sure the optimizers are only operating on their own variables:

        all_variables = tf.compat.v1.trainable_variables()
        discriminator_vars = [v for v in all_variables if v.name.startswith('discriminator/')]
        generator_vars = [v for v in all_variables if v.name.startswith('generator/')]

        discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(D_LEARNING_RATE, 0.5).minimize(
            d_loss_total, global_step=global_step, var_list=discriminator_vars)
        generator_optimizer = tf.compat.v1.train.AdamOptimizer(G_LEARNING_RATE, 0.5).minimize(
            g_loss, global_step=global_step, var_list=generator_vars)

    merged_summary = tf.compat.v1.summary.merge_all()
    # Set up a saver:
    train_writer = tf.compat.v1.summary.FileWriter(LOGDIR)

    sess = tf.compat.v1.InteractiveSession()
    if not RESTORE:
        sess.run(tf.compat.v1.global_variables_initializer())
        train_writer.add_graph(sess.graph)
        saver = tf.compat.v1.train.Saver(max_to_keep=5)
    else:
        latest_checkpoint = tf.compat.v1.train.latest_checkpoint(LOGDIR + "/checkpoints/")
        saver = tf.compat.v1.train.Saver(max_to_keep=5)
        saver.restore(sess, latest_checkpoint)





def train_step(imageset, ptcloud,index):
    ptcloud = np.transpose(ptcloud,(0,2,1))
    step = sess.run(global_step)
    #[opt_ptcloud, _, _] = sess.run([generated_ptcloud, generator_optimizer, discriminator_optimizer], feed_dict={inputImage: imageset, inputPtcloud: ptcloud})
    [opt_ptcloud, g_optimizer, encoded_img, decoded_img, acc, in_imgs] = sess.run([generated_ptcloud, generator_optimizer, encoded, decoded, total_accuracy, in_img],feed_dict={inputImage: imageset, inputPtcloud: ptcloud})
    if index % 4  == 0 or acc < 0.7:
        d_optimizer = sess.run( discriminator_optimizer,feed_dict={inputImage: imageset, inputPtcloud: ptcloud})
    #[g_optimizer] = sess.run([generator_optimizer],feed_dict={inputImage: imageset, inputPtcloud: ptcloud})
    [summary, g_l, d_l, acc_fake, acc_real, acc] = sess.run([merged_summary, g_loss, d_loss_total, accuracy_fake, accuracy_real, total_accuracy], feed_dict={inputImage: imageset, inputPtcloud: ptcloud})
    train_writer.add_summary(summary,step)
    if index != 0 and index % 200 == 0:
        tf.compat.v1.disable_eager_execution()
        saver.save(
            sess,
            LOGDIR + "/checkpoints/save",
            global_step=step)
        #index = tf.compat.v1.to_int32(step)
        with open(LOGDIR+'/output%s.pkl'%index, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([opt_ptcloud[0], g_l, d_l, acc_real,
                         acc_fake, encoded_img, decoded_img, in_imgs], f)
    index = index+1
    print('batch: ',int(step), '   loss---generator: ', g_l, ' discriminator: ',d_l, '--------------acc: ',acc)

    return opt_ptcloud[0],g_l,d_l,acc_fake,acc_real,acc,index


def train(imdataset, epochs):
    print ("Begin training ...")
    index = 0 #2563
    for epoch in range(epochs):
        start = time.time()
        for imageset_batch, ptcloud_batch in imdataset:
            # print(imageset_batch.shape,ptcloud_batch.shape)
            opt_ptcloud, g_l, d_l, acc_fake, acc_real, acc,index =train_step(imageset_batch, ptcloud_batch,index)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        print('Training in progress @ epoch %g, g_loss %g, d_loss %g accuracy %g' % (epoch, g_l, d_l, acc))
        epochs_list.append(epoch)
        gen_loss.append(g_l)
        dis_loss.append(d_l)
        true_acc.append(acc_real)
        fake_acc.append(acc_fake)
        tot_acc.append(acc)
        ptcloud_list.append(opt_ptcloud)
    with open(LOGDIR+'/objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([epochs_list,ptcloud_list, gen_loss, dis_loss,true_acc,fake_acc,tot_acc], f)


if __name__=='__main__':
    ### Set Configurations below
    DataPath = './data_all.hdf5'  # contains 100 items' images and corresponding pointclouds

    params = {'epochs': 20,           # set epochs here
            'batch_size': 20,         # set batch size here, (Future total sample number = 7268（2*2*23*79）
            'n_classes': 2,
            'width': 256,
            'height': 192,
            'n_channels': 3,
            'label': False,           #是否需要y=0的label. If 'True' ,data generator output = [img, pointcloud, y], otherwise, [img, pointcloud]
            'shuffle': True,
            'N': 2400
            } 

    ### Read data from hdf5
    with h5py.File(DataPath, 'r') as f:
        print(f.keys())       
        X1 = f['image'][:]
        X2 = f['pointcloud'][:]
        f.close()
        print('\n----------Successfully extracted data of %d items!--------------\n'%int(len(X2)),'Shape:', X1.shape, X2.shape)

    ### Split input data and create datagenerators
    X1 = X1.reshape(len(X2), params['height'], params['width'], 3)
    X1_train, X1_test, X2_train, X2_test = train_test_split(X1, X2, test_size=0.137, shuffle=False)
    X1_train = X1_train.reshape(len(X1_train) * params['height'], params['width'], 3)
    X1_test = X1_test.reshape(len(X1_test) * params['height'], params['width'], 3)
    print('TRAIN SET 2D:',X1_train.shape, '   2D:', X1_test.shape)
    print('TRAIN SET 2D:',X2_train.shape, '   3D:', X2_test.shape)

    train_batch = DataGenerator_D(X1_train, X2_train, params)    #output=[img, pointcloud, (y=0)]
    test_batch = DataGenerator_D(X1_test, X2_test, params)
    opt = set()
    train(train_batch, params['epochs'])

