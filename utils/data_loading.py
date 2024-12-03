import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2

# 图像旋转
def M_rotate_image(image , angle , cx , cy):
    '''
    图像旋转
    :param image:
    :param angle:
    :return: 返回旋转后的图像以及旋转矩阵
    '''
    (h , w) = image.shape[:2]
    # (cx , cy) = (int(0.5 * w) , int(0.5 * h))
    M = cv2.getRotationMatrix2D((cx , cy) , -angle , 1.0)
    cos = np.abs(M[0 , 0])
    sin = np.abs(M[0 , 1])

    # 计算新图像的bounding
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0 , 2] += int(0.5 * nW) - cx
    M[1 , 2] += int(0.5 * nH) - cy
    return cv2.warpAffine(image , M , (nW , nH),flags = cv2.INTER_NEAREST,borderValue = (0,0,0) ) , M


def letterbox(img, height=320, augment=False, color=(0,0,0)):
    # Resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    # resize img
    if augment:
        interpolation = np.random.choice([None, cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                          None, cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                          cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        if interpolation is None:
            img = cv2.resize(img, new_shape)
        else:
            img = cv2.resize(img, new_shape, interpolation=interpolation)
    else:
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_NEAREST)
    # print("resize time:",time.time()-s1)

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh

def img_agu_crop(img_,mask_):
    # scale_ = int(min(img_.shape[0],img_.shape[1])/15)
    scale_ = 15
    x1 = max(0,random.randint(0,scale_))
    y1 = max(0,random.randint(0,scale_))
    x2 = min(img_.shape[1]-1,img_.shape[1] - random.randint(0,scale_))
    y2 = min(img_.shape[0]-1,img_.shape[1] - random.randint(0,scale_))
    # print(img_.shape,'-crop- : ',x1,y1,x2,y2)
    try:
        img_crop_ = img_[y1:y2,x1:x2,:]
        mask_crop_ = mask_[y1:y2,x1:x2]
    except:
        img_crop_ = img_
        mask_crop_ = mask_
        print("img_agu_crop error ")
    return img_crop_,mask_crop_

def img_agu_channel_same(img_):
    img_a = np.zeros(img_.shape, dtype = np.uint8)
    gray = cv2.cvtColor(img_,cv2.COLOR_RGB2GRAY)
    img_a[:,:,0] =gray
    img_a[:,:,1] =gray
    img_a[:,:,2] =gray

    return img_a

# 图像亮度、对比度增强
def contrast_img(img, c, b):  # 亮度就是每个像素所有通道都加上b
    rows, cols, channels = img.shape
    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, channels])
    dst = cv2.addWeighted(img.astype(np.float), c, blank, 1-c, b)
    dst = np.clip(dst,0,255).astype(np.uint8)
    return dst

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '',flag_agu = True):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        self.agu = flag_agu

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH))
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:

            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        if random.random()>0.7 and self.agu:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT) #左右对换。
            img = img.transpose(Image.FLIP_LEFT_RIGHT)  # 上下对换。
        if random.random()>0.7 and self.agu:
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM) #左右对换。
            img = img.transpose(Image.FLIP_TOP_BOTTOM)  # 上下对换。
        #---------------
        mask_array = np.array(mask)
        img_array = np.array(img)
        img_ = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        #----- 数据增强
        if random.random() > 0.5 and self.agu:
            c = float(random.randint(80,120))/100.
            b = random.randint(-55,40)
            img_ = contrast_img(img_, c, b)
        if random.random() > 0.8 and self.agu:
            img_hsv=cv2.cvtColor(img_,cv2.COLOR_BGR2HSV)
            hue_x = random.randint(-10,10)

            img_hsv[:,:,0]=(img_hsv[:,:,0]+hue_x)
            img_hsv[:,:,0] =np.maximum(img_hsv[:,:,0],0)
            img_hsv[:,:,0] =np.minimum(img_hsv[:,:,0],180)#范围 0 ~180
            img_=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
        if random.random() > 0.888888889 and self.agu:
            img_ = img_agu_channel_same(img_)
        
        if random.random() > 0.7 and self.agu:
            cx = int(img_.shape[1]/2)
            cy = int(img_.shape[0]/2)
            angle = random.randint(-180,180)
            range_limit_x = int(min(6,img_.shape[1]/16))
            range_limit_y = int(min(6,img_.shape[0]/16))
            offset_x = random.randint(-range_limit_x,range_limit_x)
            offset_y = random.randint(-range_limit_y,range_limit_y)
            if not(angle==0 and offset_x==0 and offset_y==0):
                img_,_  = M_rotate_image(img_ , angle , cx+offset_x , cy+offset_y)
                mask_array,_  = M_rotate_image(mask_array , angle , cx+offset_x , cy+offset_y)
        #----- 数据增强
        #img_array = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        img_array = img_
        if random.random() > 0.5 and self.agu:
            img_array,mask_array = img_agu_crop(img_array,mask_array)
        if False:
            img_array,_,_,_ = letterbox(img_array,height=128)
            mask_array,_,_,_ = letterbox(mask_array,height=128)
        else:
            img_array = cv2.resize(img_array, (128,128), interpolation = random.randint(0,5))
            mask_array = cv2.resize(mask_array, (128,128), interpolation = cv2.INTER_NEAREST)
 
        img = Image.fromarray(np.uint8(img_array))
        mask = Image.fromarray(np.uint8(mask_array))
        #---------------
        # print("mask",mask.size)
        # print(mask_file[0],img_file[0])

        #hhidx = np.where(np.array(mask) !=0)
        # print(np.array(mask)[hhidx])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1,flag_agu = True):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
