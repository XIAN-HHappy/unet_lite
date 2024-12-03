# -*-coding: utf-8 -*-

import os, sys
sys.path.append(os.getcwd())
import onnxruntime
import onnx
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

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
class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image_numpy):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        output = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return output
if __name__ == "__main__":
    img_size = 128
    # model = ONNXModel("unet_{}.onnx".format(img_size))
    model = ONNXModel("unet_128-2.7M.onnx")
    path_ = "./images/"
    for f_ in os.listdir(path_):

        img0 = cv2.imread(path_ + f_)
        img = cv2.resize(img0, (img_size,img_size), interpolation = cv2.INTER_CUBIC)
        # img,_,_,_ = letterbox(img, height=img_size, augment=False, color=(0,0,0))
        # img_ndarray = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_ndarray = img.transpose((2, 0, 1))
        img_ndarray = img_ndarray / 255.
        img_ndarray = np.expand_dims(img_ndarray, 0)

        output = model.forward(img_ndarray.astype('float32'))[0]
        mask_fg = (output[0,1]*255).astype(np.uint8)
        mask_bg = (output[0,0]*255).astype(np.uint8)

        finger_mask_ = np.zeros((img.shape[0],img.shape[1])).astype(np.uint8)
        finger_mask_ = np.where((mask_fg>150),255,0).astype(np.uint8)
        # print("output : \n",output.shape)
        cv2.namedWindow("mask",0)
        cv2.imshow("mask",finger_mask_)
        cv2.namedWindow("img",0)
        cv2.imshow("img",img0)
        cv2.namedWindow("mask_fg",0)
        cv2.imshow("mask_fg",mask_fg)
        cv2.namedWindow("mask_bg",0)
        cv2.imshow("mask_bg",mask_bg)
        
        cv2.waitKey(0)
