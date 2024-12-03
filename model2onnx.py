import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./checkpoints/w_checkpoint_epoch999.pth"
model= torch.load(model_path)
input_size = 128
batch_size = 1  #批处理大小
input_shape = (3, input_size,input_size)   #输入数据,改成自己的输入shape
print("input_size : ",input_size)
print("model_path : ",model_path)
# #set the model to inference mode
model.eval()

x = torch.randn(batch_size, *input_shape)   # 生成张量
x = x.to(device)
export_onnx_file = "unet_{}.onnx".format(input_size)		# 目的ONNX文件名
torch.onnx.export(model,
                    x,
                    export_onnx_file,
                    opset_version=9,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],	# 输入名
                    output_names=["output"],	# 输出名
                    #dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
                    #                "output":{0:"batch_size"}}
                    )
