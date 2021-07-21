# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
# %%
import cv2
import numpy as np
import os
import random
import skimage.io as io
from IPython import get_ipython
from pycocotools.coco import COCO

### For visualizing the outputs ###
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# get_ipython().run_line_magic('matplotlib', 'inline')



# %%
annFile = "/ldap_home/jincheng.lyu/project/SOLO/tools/convert_datasets/instances_train.json"
coco = COCO(annFile)


# %%
# Define the classes (out of the 81) which you want to see. Others will not be shown.
# filterClasses = ['Air conditioner', 'Watches', 'Biscuit', 'Rope']
filterClasses = ['Air conditioner']

# Fetch class IDs only corresponding to the filterClasses
catIds = coco.getCatIds(catNms=filterClasses) 

# Get all images containing the above Category IDs
imgIds = coco.getImgIds(catIds=catIds)
print("Number of images containing all the  classes:", len(imgIds))


# %%
def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


# %%
imgIds = coco.getImgIds(catIds=[0])
imgIds


# %%
# dataDir = "/ldap_home/jincheng.lyu/data/product_segmentation/synthetics_cocoformat/train/"
# # load and display a random image
# img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
# I = io.imread('{}/images/{}/{}'.format(dataDir, dataType, img['file_name']))/255.0

# plt.axis('off')
# plt.imshow(I)
# plt.show()


