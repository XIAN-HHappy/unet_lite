# UNet-Lite
unet , pytorch  

# 使用
* step1： 准备数据集 data
* step2： 训练 python train.py
* step3:  转 onnx 模型，注意推理时需要将 unet/unet_model.py 脚本 ,的 sigmoid 操作打开，否则推理需要额外加sigmoid操作。运行脚本 python model2onnx.py 。
* step4:  进入 onnx_test文件夹，设置 onnx_inference.py 模型和图片路径，运行脚本 python onnx_inference.py