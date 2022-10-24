#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from hlmobilenetv2 import hlmobilenetv2
import os
from time import time
from PIL import Image
from collections import OrderedDict
from tqdm import tqdm
# ignore warnings
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import shutil



pathL = Path("../jpg/rootdata").glob('**/*.jpg')
name_dic={}
for i,path in enumerate(pathL):
    shutil.copy("../jpg/rootdata/"+str(path.name), '../jpg/name_trans/'+str(i)+'.jpg')
    name_dic[i]=path.name


pathlist = Path("../jpg/name_trans").glob('**/*.jpg')
#myfile = 'filename.txt'

path_name=[]
for path in pathlist:
    path_name.append(str(path.name))

print(str(len(path_name))+'件のファイルに処理を適応します。')
#print(path_name)
print('\n－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－\n')
print('輪郭の検出処理中…')
for i in tqdm(range(len(path_name))):
    image_path = path_name[i]
    img = cv2.imread('../jpg/name_trans/'+path_name[i])
    w_file_1='../jpg/img/'+str(path_name[i])+'.png'
    cv2.imwrite(w_file_1.replace('.jpg',''),img)
    img = img[...,::-1] #BGR->RGB
    h,w,_ = img.shape
    img = cv2.resize(img,(320,320))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    model = model.to(device)
    model.eval();
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output = output.argmax(0)
    mask = output.byte().cpu().numpy()
    mask = cv2.resize(mask,(w,h))
    img = cv2.resize(img,(w,h))
    #plt.gray()
    #plt.figure(figsize=(20,20))
    #plt.subplot(1,2,1)
    #plt.imshow(img)
    #plt.subplot(1,2,2)
    #plt.imshow(mask);
    
    def gen_trimap(mask,k_size=(5,5),ite=1):
        kernel = np.ones(k_size,np.uint8)
        eroded = cv2.erode(mask,kernel,iterations = ite)
        dilated = cv2.dilate(mask,kernel,iterations = ite)
        trimap = np.full(mask.shape,128)
        trimap[eroded >= 1] = 255
        trimap[dilated == 0] = 0
        return trimap
    
    trimap = gen_trimap(mask,k_size=(10,10),ite=5)
    w_file_2='../jpg/trimap/'+str(path_name[i])+'.png'
    cv2.imwrite(w_file_2.replace('.jpg',''),trimap)
    #plt.figure(figsize=(20,20))
    #plt.subplot(1,2,1)
    #plt.imshow(img)
    #plt.subplot(1,2,2)
    #plt.imshow(trimap)


# In[2]:


print('\n－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－\n')
print('学習モデルから輪郭の細部を検出中…('+str(len(path_name))+'件)')

IMG_SCALE = 1./255
IMG_MEAN = np.array([0.485, 0.456, 0.406, 0]).reshape((1, 1, 4))
IMG_STD = np.array([0.229, 0.224, 0.225, 1]).reshape((1, 1, 4))

STRIDE = 32
RESTORE_FROM = 'indexnet_matting.pth.tar'
#RESTORE_FROM = 'deep_matting.pth.tar'
RESULT_DIR = '../jpg/mattes'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# load pretrained model
net = hlmobilenetv2(
        pretrained=False,
        freeze_bn=True, 
        output_stride=STRIDE,
        apply_aspp=True,
        conv_operator='std_conv',
        decoder='indexnet',
        decoder_kernel_size=5,
        indexnet='depthwise',
        index_mode='m2o',
        use_nonlinear=True,
        use_context=True
    )

try:
    checkpoint = torch.load(RESTORE_FROM, map_location=device)
    pretrained_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if 'module' in key:
            key = key[7:]
        pretrained_dict[key] = value
except:
    raise Exception('Please download the pretrained model!')
net.load_state_dict(pretrained_dict)
net.to(device)
if torch.cuda.is_available():
    net = nn.DataParallel(net)

# switch to eval mode
net.eval()

def read_image(x):
    img_arr = np.array(Image.open(x))
    return img_arr

def image_alignment(x, output_stride, odd=False):
    imsize = np.asarray(x.shape[:2], dtype=np.float)
    if odd:
        new_imsize = np.ceil(imsize / output_stride) * output_stride + 1
    else:
        new_imsize = np.ceil(imsize / output_stride) * output_stride
    h, w = int(new_imsize[0]), int(new_imsize[1])

    x1 = x[:, :, 0:3]
    x2 = x[:, :, 3]
    new_x1 = cv2.resize(x1, dsize=(w,h), interpolation=cv2.INTER_CUBIC)
    new_x2 = cv2.resize(x2, dsize=(w,h), interpolation=cv2.INTER_NEAREST)

    new_x2 = np.expand_dims(new_x2, axis=2)
    new_x = np.concatenate((new_x1, new_x2), axis=2)

    return new_x

def inference(image_path, trimap_path):
    with torch.no_grad():
        image, trimap = read_image(image_path), read_image(trimap_path)
        trimap = np.expand_dims(trimap, axis=2)
        image = np.concatenate((image, trimap), axis=2)
        
        h, w = image.shape[:2]

        image = image.astype('float32')
        image = (IMG_SCALE * image - IMG_MEAN) / IMG_STD
        image = image.astype('float32')

        image = image_alignment(image, STRIDE)
        inputs = torch.from_numpy(np.expand_dims(image.transpose(2, 0, 1), axis=0))
        inputs = inputs.to(device)
        
        # inference
        start = time()
        outputs = net(inputs)
        end = time()

        outputs = outputs.squeeze().cpu().numpy()
        alpha = cv2.resize(outputs, dsize=(w,h), interpolation=cv2.INTER_CUBIC)
        alpha = np.clip(alpha, 0, 1) * 255.
        trimap = trimap.squeeze()
        mask = np.equal(trimap, 128).astype(np.float32)
        alpha = (1 - mask) * trimap + mask * alpha

        _, image_name = os.path.split(image_path)
        Image.fromarray(alpha.astype(np.uint8)).save(os.path.join(RESULT_DIR, image_name))
        # Image.fromarray(alpha.astype(np.uint8)).show()

        running_frame_rate = 1 * float(1 / (end - start)) # batch_size = 1
        #print('framerate: {0:.2f}Hz'.format(running_frame_rate))

pathlist_1= Path("../jpg/img").glob('**/*.png')
pathlist_2= Path("../jpg/trimap").glob('**/*.png')

img_names=[]
trimap_names=[]
for path in pathlist_1:
    img_names.append('../jpg/img/'+str(path.name))
for path in pathlist_2:
    trimap_names.append('../jpg/trimap/'+str(path.name))

if __name__=='__main__':
    image_path = img_names
    trimap_path = trimap_names
    
    for image, trimap in tqdm(zip(image_path, trimap_path),total=len(image_path)):
        inference(image, trimap)
#print(img_names)
#print(trimap_names)


# In[ ]:


pathlist_3= Path("../jpg/mattes").glob('**/*.png')
matte_names=[]
for path in pathlist_3:
    matte_names.append('../jpg/mattes/'+str(path.name))


print('\n－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－\n')
print('画像の切り抜き処理中…('+str(len(path_name))+'件)')
for j in tqdm(range(len(img_names))):
    img = cv2.imread(img_names[j])
    img = img[...,::-1]
    matte = cv2.imread(matte_names[j])
    h,w,_ = img.shape
    bg = np.full_like(img,255) #white background
    
    img = img.astype(float)
    bg = bg.astype(float)
    
    matte = matte.astype(float)/255
    img = cv2.multiply(img, matte)
    bg = cv2.multiply(bg, 1.0 - matte)
    outImage = cv2.add(img, bg)
    outImage=(outImage/255)
    Img = np.clip(outImage * 255, 0, 255).astype(np.uint8)
    
    Key=int(img_names[j].replace('../jpg/img/','').replace('.png',''))
    reN=name_dic[Key]
    
    plt.imsave('../jpg/結果/'+reN.replace('../jpg/img/', '').replace('.jpg', '').replace('.png', '') +'.jpg',Img)
    #plt.imshow(Img)


# In[ ]:




