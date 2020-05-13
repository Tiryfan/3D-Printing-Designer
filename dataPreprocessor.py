from sklearn.model_selection import train_test_split
from torchvision import transforms
import tensorflow
from PIL import Image
import os.path as osp
import cv2
import os
import h5py
import random
import numpy as np
import scipy.io
from tqdm import tqdm
import json
import matplotlib.pyplot as plt


##### Create 2d dataset
class Processor:
    ### Read pointcloud data into np.array
    def readXYZfile(self, filename, Separator):
        data = np.empty(shape=[0, 3], dtype=float)
        f = open(filename, 'r')
        line = f.readline()
        num = 0
        while line:  # read lines
            c, d, e = line.split(Separator)
            c, d, e = float(c), float(d), float(e)  # transform from string to float
            data = np.append(data, [[c, d, e]], axis=0)
            num = num + 1
            line = f.readline()
        f.close()
        return data, num

    ### Store pointcloud data into json {'item id': value}
    def storePointcloud(self, dataset, out_dir, points_need):
            uselessfile = osp.join(dataset, '.DS_Store')
            if os.path.exists(uselessfile):
                os.remove(uselessfile)
                print('<.DS_store> File Removed!')
            x = {}
            items = os.listdir(dataset)

            count = 0
            efficient_model = 0
            for i in tqdm(items):
                path_Item = osp.join(dataset, i)
                fsize = os.path.getsize(path_Item)
                if (fsize <= 1000000):
                    pcId = i[:-4]
                    single, points = self.readXYZfile(path_Item, ' ')
                    if (points >= points_need):
                        slice_points = points - points_need
                        single = np.delete(single, slice(0, slice_points), axis=0)   # (N, 3)
                        single = single.T               # transpose to (3, N)
                        single = single.reshape(-1)     # reshape to (1, 3N)
                        # print(single.shape)
                        single = single.tolist()
                        x[pcId] = single
                        count = count + 1
            # print(num_points)
            print("efficient_model", count)
            jsObj = json.dumps(x)
            with open(out_dir, 'w') as f:
                f.write(jsObj)
            f.close()
            print('\n------------Successfully created a json file with point-clouds!--------------\n')
            return x, count

    ### Store RGB images as grayscale images with correspongding pointcloud as np.array into an hdf5 file 
    def storeIMG(self, narrowdown, dataset, file_img, file_pointcloud, N):
        uselessfile = osp.join(dataset, '.DS_Store')
        if os.path.exists(uselessfile):
            os.remove(uselessfile)
            print('<.DS_store> File Removed!')
        
        # read pointcloud data with item id
        with open(file_pointcloud, 'r') as f1:
            pointcloud = json.load(f1)

        items = os.listdir(dataset)
        x1 = np.empty((0, 256, 3)) 
        x2 = np.empty((0, 3 * N)) 
        for itemID in tqdm(items):
            x2_single = pointcloud.get(itemID, -1)
            if (x2_single == -1):
                continue

            path_Item = osp.join(dataset, itemID)
            views = os.listdir(path_Item)
            index = []

            x1_single = np.empty((0,256))         #img of 1 item (192, 256)
            x2_single = np.array(x2_single).reshape(1, 3 * N)       #pointcloud (1,3N)
            num = len(views)
        
            for j in range(3):
                if narrowdown == True:
                    index.append(random.randrange(num//3 * j, num//3 * (j+1)))
                else:
                    index.append(j)
                view = cv2.imread(osp.join(path_Item, views[index[j]]))
                gview = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
                x1_single = np.vstack((x1_single, gview))
            x1_single = x1_single.reshape(192, 256, 3)
            x1 = np.vstack((x1, x1_single))
            x2 = np.vstack((x2, x2_single))
            # print(itemID, '  ', x1.shape, x2.shape)
        print(x1.shape, x2.shape)
        
        with h5py.File(file_img, 'w') as f2:
            f2.create_dataset('image', data = x1)
            f2.create_dataset('pointcloud', data = x2)
        print('\n------------Successfully created a hdf5 file with image & pointcloud!-------------\n')
        

    ###read images and xyz files for testing
    def test(self, path_image, path_pointcloud, NARROWDOWN, REAL3D, width, num_views, num_points):
        # Delete useless files
        uselessfile = osp.join(path_image, '.DS_Store')
        if os.path.exists(uselessfile):
            os.remove(uselessfile)
            print('<.DS_store> File Removed!')

        views = os.listdir(path_image)
        x1 = np.empty((0, width, num_views)) 
        x1 = np.empty((0, width))                    #one view (192, 256)
        index = []
        num = len(views)
        i = 1
        for j in range(num_views):
            if NARROWDOWN:
                index.append(random.randrange(num//3 * j, num//3 * (j+1)))
            else:
                index.append(j)

            view = cv2.imread(osp.join(path_image, views[index[j]]))
            title = "Before" + str(j+1)
            plt.subplot(3, 2, i)
            plt.imshow(view)
            plt.title(title,fontsize=8)
            plt.xticks([])
            plt.yticks([])
            i = i + 1

            gray_view = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
            title = "Input" + str(j+1)
            plt.subplot(3, 2, i)
            plt.imshow(gray_view, cmap="gray")
            plt.title(title,fontsize=8)
            plt.xticks([])
            plt.yticks([])
            i = i + 1

            x1 = np.vstack((x1, gray_view))
        plt.show()
        # plt.savefig("input_images.png")
        x1 = x1.reshape(192, 256, 3)
        if REAL3D:
            x2 = self.readXYZfile(path_pointcloud, ' ')
            x2 = np.array(x2).reshape(1, 3 * num_points)       #pointcloud (1,3N)
            print(x1.shape, x2.shape)
            return x1, x2
        return x1




##### Only for testing generator model with single output [img]
class DataGenerator_G(tensorflow.keras.utils.Sequence):
    def __init__(self, dataset, params):
        self.batch_size = params['batch_size']
        self.n_classes = params['n_classes']
        self.shuffle = params['shuffle']
        self.height = params['height']
        self.width = params['width']
        self.n_channels = params['n_channels']
        self.label = params['label']
        self.dim = (params['height'], params['width'], params['n_channels'])
        self.X = dataset
        # self.X, self.transform = dataset
        self.n_items = int(len(self.X) // self.height)
        self.on_epoch_end()

    ### Total number of batches per epoch
    def __len__(self): 
        n_batch = int(self.n_items // self.batch_size)
        return n_batch

    ### Generate one batch of data
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]
        X, Y = self.__data_generation(indexes)
        if self.label == True:
            return X, Y
        return X

    ### Generates data containing batch_size samples
    def __data_generation(self, indexes):
        X = np.empty((self.batch_size, * self.dim)) #(0,192,256,3)
        Y = np.zeros((self.batch_size), dtype=int)  # y = 0, fake
        i = 0
        for idx in indexes:
            img = self.X[idx * self.height : (idx+1) * self.height,:,:]
            X[i,] = img
            i = i + 1
        return X, Y

    ### Updates indexes after each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(self.n_items)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


##### output=[img, pointcloud], img = (B, 192, 256,3), pointcloud = (B, N=2304, 3)
class DataGenerator_D(tensorflow.keras.utils.Sequence):
    def __init__(self, dataset_img, dataset_pc, params):
        self.X1 = np.array(dataset_img)
        self.X2 = np.array(dataset_pc)
        self.dim1 = (params['height'], params['width'], params['n_channels'])
        self.dim2 = (params['N'], 3)
        self.batch_size = params['batch_size']
        self.n_classes = params['n_classes']
        self.shuffle = params['shuffle']
        self.height = params['height']
        self.width = params['width']
        self.n_channels = params['n_channels']
        self.label = params['label']
        self.n_items = int(len(self.X2))
        self.on_epoch_end()

    ### Total number of batches per epoch
    def __len__(self): 
        n_batch = int(self.n_items // self.batch_size)
        return n_batch

    ### Generate one batch of data
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]
        X1, X2, Y = self.__data_generation(indexes)
        if self.label == True:
            return X1, X2, Y
        return X1, X2

    ### Generates data containing batch_size samples
    def __data_generation(self, indexes):
        # print(self.X1.shape,self.X2.shape)
        X1 = np.empty((self.batch_size, * self.dim1)) #(0,192,256,3)
        X2 = np.empty((self.batch_size, * self.dim2)) #(0,N,3)
        Y = np.zeros((self.batch_size), dtype=int)  # y = 0, fake
        i = 0
        for idx in indexes:
            img = self.X1[idx * self.height : (idx+1) * self.height,:,:]
            pointcloud = self.X2[idx : idx+1,:].reshape(1, *self.dim2)
            X1[i,] = img
            X2[i,] = pointcloud
            i = i + 1
        return  X1, X2, Y

    ### Updates indexes after each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(self.n_items)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


