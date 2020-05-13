from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
import scipy.misc, scipy.io
import time, os, sys
import threading
import tensorflow as tf

def set():
    # parse input arguments
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    # these stay fixed
    opt.group = 0
    opt.std = 0.1
    opt.outViewN = 8
    opt.novelN = 5
    opt.arch = None
    opt.batchSize = 20
    opt.lr = 1e-4
    opt.sampleN = 100
    opt.BNepsilon = 1e-5
    opt.BNdecay = 0.999
    opt.inputViewN = 24
    opt.inSize = "192x256"
    opt.outSize = '15x20'
    # ------ below automatically set ------
    opt.training = True
    opt.visBlockSize = int(np.floor(np.sqrt(opt.batchSize)))
    opt.fuseTrans = np.load("trans_fuse{0}.npy".format(opt.outViewN))
    # opt.fuseTrans = np.load("/Users/ellie/Desktop/project/599_project_code_backup/trans_fuse{0}.npy".format(opt.outViewN))
    opt.inH,opt.inW = [int(x) for x in opt.inSize.split("x")]
    opt.outH,opt.outW = [int(x) for x in opt.outSize.split("x")]
    opt.H,opt.W = [int(x) for x in opt.outSize.split("x")]
    opt.renderDepth = 0.0
    opt.Khom3Dto2D = np.array([[opt.W,0 ,0,opt.W/2],
                               [0,-opt.H,0,opt.H/2],
                               [0,0,-1,0],
                               [0,0, 0,1]],dtype=np.float32)
    opt.Khom2Dto3D = np.array([[opt.outW,0 ,0,opt.outW/2],
                               [0,-opt.outH,0,opt.outH/2],
                               [0,0,-1,0],
                               [0,0, 0,1]],dtype=np.float32)
    return opt

def quaternionToRotMatrix(q):
	with tf.name_scope("quaternionToRotMatrix"):
		qa,qb,qc,qd = tf.unstack(q,axis=1)
		R = tf.transpose(tf.stack([[1-2*(qc**2+qd**2),2*(qb*qc-qa*qd),2*(qa*qc+qb*qd)],
								   [2*(qb*qc+qa*qd),1-2*(qb**2+qd**2),2*(qc*qd-qa*qb)],
								   [2*(qb*qd-qa*qc),2*(qa*qb+qc*qd),1-2*(qb**2+qc**2)]]),perm=[2,0,1])
	return R

def transParamsToHomMatrix(q,t):
    with tf.name_scope("transParamsToHomMatrix"):
        N = tf.shape(q)[0]
        R = quaternionToRotMatrix(q)
        Rt = tf.concat([R,tf.expand_dims(t,-1)],axis=2)
        hom_aug = tf.concat([tf.zeros([N,1,3]),tf.ones([N,1,1])],axis=2)
        RtHom = tf.concat([Rt,hom_aug],axis=1)
    return RtHom

def get3DhomCoord(XYZ,opt):
    with tf.name_scope("get3DhomCoord"):
        ones = tf.ones([opt.batchSize, opt.outViewN, opt.outH, opt.outW])
        XYZhom = tf.transpose(tf.reshape(tf.concat([XYZ, ones], axis=1), [opt.batchSize, 4, opt.outViewN, -1]),
                              perm=[0, 2, 1, 3])
    return XYZhom # [B,V,4,HW]


def fuse3D(opt,XYZ,fuseTrans): # [B,H,W,3V],[B,H,W,V]
    with tf.name_scope("transform_fuse3D"):
        XYZ = tf.transpose(XYZ,perm=[0,3,1,2]) # [B,3V,H,W]
        # 2D to 3D coordinate transformation
        invKhom = np.linalg.inv(opt.Khom2Dto3D)
        invKhomTile = np.tile(invKhom,[opt.batchSize,opt.outViewN,1,1])
        # viewpoint rigid transformation
        q_view = fuseTrans
        t_view = np.tile([0,0,-opt.renderDepth],[opt.outViewN,1]).astype(np.float32)
        RtHom_view = transParamsToHomMatrix(q_view,t_view)
        RtHomTile_view = tf.tile(tf.expand_dims(RtHom_view,0),[opt.batchSize,1,1,1])
        invRtHomTile_view = tf.linalg.inv(RtHomTile_view)
        # effective transformation
        RtHomTile = tf.matmul(invRtHomTile_view,invKhomTile) # [B,V,4,4]
        RtTile = RtHomTile[:,:,:3,:] # [B,V,3,4]
        # transform depth stack
        XYZhom = get3DhomCoord(XYZ,opt) # [B,V,4,HW]
        XYZid = tf.matmul(RtTile,XYZhom) # [B,V,3,HW]
        # fuse point clouds
        XYZid = tf.reshape(tf.transpose(XYZid,perm=[0,2,1,3]),[opt.batchSize,3,-1]) # [B,3,VHW]
    return XYZid # [B,3,VHW]


def encoder(opt,image,i): # [B,H,W,3]
    def conv2Layer(opt,feat,outDim):
        weight,bias = createVariable(opt,[3,3,int(feat.shape[-1]),outDim])
        conv = tf.nn.conv2d(feat,weight,strides=[1,2,2,1],padding="SAME")+bias
        batchnorm = batchNormalization(opt,conv,i,type="conv")
        relu = tf.nn.relu(batchnorm)
        return relu
    def linearLayer(opt,feat,outDim,final=False):
        weight,bias = createVariable(opt,[int(feat.shape[-1]),outDim])
        fc = tf.matmul(feat,weight)+bias
        batchnorm = batchNormalization(opt,fc,i,type="fc")
        relu = tf.nn.relu(batchnorm)
        return relu if not final else fc

    with tf.compat.v1.variable_scope("encoder"):
        feat = image
        with tf.compat.v1.variable_scope("conv1"): feat = conv2Layer(opt, feat, 24)  # 96x128
        with tf.compat.v1.variable_scope("conv2"): feat = conv2Layer(opt, feat, 48)  # 48x64
        with tf.compat.v1.variable_scope("conv3"): feat = conv2Layer(opt, feat, 96)  # 24x32
        with tf.compat.v1.variable_scope("conv4"): feat = conv2Layer(opt, feat, 192)  # 12x16
        feat = tf.reshape(feat,[opt.batchSize,12*16*feat.shape[-1]])
        with tf.compat.v1.variable_scope("fc1"): feat = linearLayer(opt, feat, 2048)
        with tf.compat.v1.variable_scope("fc2"): feat = linearLayer(opt, feat, 1024)
        with tf.compat.v1.variable_scope("fc3"): feat = linearLayer(opt, feat, 512, final=True)
        latent = feat
    return latent

# build decoder
def decoder(opt,latent,i):
    def linearLayer(opt,feat,outDim):
        weight,bias = createVariable(opt,[int(feat.shape[-1]),outDim])
        fc = tf.matmul(feat,weight)+bias
        batchnorm = batchNormalization(opt,fc,i,type="fc")
        relu = tf.nn.relu(batchnorm)
        return relu
    def deconv2Layer(opt,feat,outDim):
        weight,bias = createVariable(opt,[3,3,int(feat.shape[-1]),outDim],stddev=opt.std)
        conv = tf.nn.conv2d(feat,weight,strides=[1,1,1,1],padding="SAME")+bias
        batchnorm = batchNormalization(opt,conv,i,type="conv")
        relu = tf.nn.relu(batchnorm)
        return relu
    def pixelconv2Layer(opt,feat,outDim):
        H, W = int(feat.shape[1]), int(feat.shape[2])
        resize = tf.image.resize(feat, [int(H * 5), int(W * 5)], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        weight,bias = createVariable(opt,[1,1,int(resize.shape[-1]),outDim],gridInit=True)
        conv = tf.nn.conv2d(resize,weight,strides=[1,1,1,1],padding="SAME")+bias
        return conv

    # def run(opt,feat, outDim):
    with tf.compat.v1.variable_scope("decoder"):
        feat = tf.nn.relu(latent)
        with tf.compat.v1.variable_scope("fc1"): feat = linearLayer(opt, feat, 960)
        with tf.compat.v1.variable_scope("fc2"): feat = linearLayer(opt, feat, 1920)
        with tf.compat.v1.variable_scope("fc3"): feat = linearLayer(opt, feat, 3840)
        feat = tf.reshape(feat, [opt.batchSize, 3, 4, -1])
        with tf.compat.v1.variable_scope("deconv1"): feat = deconv2Layer(opt, feat, 256)  # 8x8
        with tf.compat.v1.variable_scope("deconv2"): feat = deconv2Layer(opt, feat, 192)  # 16x16
        with tf.compat.v1.variable_scope("deconv3"): feat = deconv2Layer(opt, feat, 96)  # 32x32
        with tf.compat.v1.variable_scope("deconv4"): feat = deconv2Layer(opt, feat, 48)  # 64x64
        with tf.compat.v1.variable_scope("deconv5"): feat = deconv2Layer(opt, feat, 24)  # 128x128
        with tf.compat.v1.variable_scope("pixelconv"): feat = pixelconv2Layer(opt, feat, opt.outViewN * 3)  # 128x128
        XYZ= feat # [B,H,W,3V]-->[B,H,W,V]
    return XYZ # [B,H,W,3V]    


# auxiliary function for creating weight and bias
def createVariable(opt,weightShape,biasShape=None,stddev=None,gridInit=False):
    if biasShape is None: biasShape = [weightShape[-1]]
    weight = tf.Variable(tf.random.normal(weightShape,stddev=opt.std),dtype=np.float32,name="weight")
    if gridInit:
        X,Y = np.meshgrid(range(opt.outW),range(opt.outH),indexing="xy") # [H,W]
        X,Y = X.astype(np.float32),Y.astype(np.float32)
        initTile = np.concatenate([np.tile(X,[opt.outViewN,1,1]),
                                   np.tile(Y,[opt.outViewN,1,1]),
                                   np.zeros([opt.outViewN,opt.outH,opt.outW],dtype=np.float32)],axis=0) # [4V,H,W]
        biasInit = np.expand_dims(np.transpose(initTile,axes=[1,2,0]),axis=0) # [1,H,W,4V]
    else:
        biasInit = tf.constant(0.0,shape=biasShape)
    bias = tf.Variable(biasInit,dtype=np.float32,name="bias")
    return weight,bias

# batch normalization wrapper function
def batchNormalization(opt,input,i,type):
    with tf.compat.v1.variable_scope("batchNorm_{}".format(i)):
        globalMean = tf.compat.v1.get_variable("mean", shape=[input.shape[-1]], dtype=tf.float32, trainable=False,
                                    initializer=tf.constant_initializer(0.0))
        globalVar = tf.compat.v1.get_variable("var", shape=[input.shape[-1]], dtype=tf.float32, trainable=False,
                                    initializer=tf.constant_initializer(1.0))
    #globalMean = tf.Variable(tf.zeros(input.shape[-1]),dtype=tf.float32,trainable=False)
    #globalVar = tf.Variable(tf.ones(input.shape[-1]),dtype=tf.float32,trainable=False)
        if opt.training:
            if type=="conv": batchMean,batchVar = tf.nn.moments(input,axes=[0,1,2])
            elif type=="fc": batchMean,batchVar = tf.nn.moments(input,axes=[0])
            trainMean = tf.compat.v1.assign_sub(globalMean,(1-opt.BNdecay)*(globalMean-batchMean))
            trainVar = tf.compat.v1.assign_sub(globalVar,(1-opt.BNdecay)*(globalVar-batchVar))
            with tf.control_dependencies([trainMean,trainVar]):
                output = tf.nn.batch_normalization(input,batchMean,batchVar,None,None,opt.BNepsilon)
        else: output = tf.nn.batch_normalization(input,globalMean,globalVar,None,None,opt.BNepsilon)
    return output

