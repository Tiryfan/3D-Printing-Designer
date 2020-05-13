import pickle
import numpy as np
import matplotlib.pyplot as plt

# select result of trian/test
TRAIN = False
TEST = True
 
if TRAIN:
    path_train_out = './save_train/xxxxxx.pkl'
    with open(path_train_out,'rb') as f:  # Python 3: open(..., 'rb')
        ptcloud_list, gen_loss, dis_loss, true_acc, fake_acc, encoded_img, decoded_img,input_img = pickle.load(f)


if TEST:
    path_test_out = './save_test/xxxxxx.pkl'
    with open(path_test_out,'rb') as f:  
        ptcloud_list, gen_loss, dis_loss, true_acc, fake_acc, encoded_img, decoded_img, input_img = pickle.load(f)




### store numpy ararry as txt file
ptcloud_list = ptcloud_list.T
print(ptcloud_list.shape)
path_pointclouds = './output.txt'
mat = np.matrix(ptcloud_list)
with open(path_pointclouds, 'wb') as f:
    for line in mat:
        np.savetxt(f, line, fmt='%.2f')

### show input images
print(input_img.shape)
for item in input_img:
    i = 1
    for gray_view in item:
        title = "gray scaled input"
        plt.subplot(1, 3, i)
        plt.imshow(gray_view, cmap="gray")
        plt.title(title,fontsize=8)
        plt.xticks([])
        plt.yticks([])
        i = i + 1
    plt.show()
    