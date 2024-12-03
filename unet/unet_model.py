""" Full assembly of the parts to form the complete network """

from .unet_parts import *
# from unet_parts import *

#
# class UNet(nn.Module): # 7.4 M
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         model_scale = 0
#         self.inc = DoubleConv(n_channels, 16)
#         self.down1 = Down(16, 32)
#         self.down2 = Down(32, 64)
#         self.down3 = Down(64, 128)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(128, 256 // factor)
#         self.up1 = Up(256, 128 // factor, bilinear)
#         self.up2 = Up(128, 64 // factor, bilinear)
#         self.up3 = Up(64, 32 // factor, bilinear)
#         self.up4 = Up(32, 16, bilinear)
#         self.outc = OutConv(16, n_classes)
#
#

# class UNet(nn.Module): # 5.2 M
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         model_scale = 0
#         self.inc = DoubleConv(n_channels, 16)
#         self.down1 = Down(16, 32)
#         self.down2 = Down(32, 64)
#         self.down3 = Down(64, 96)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(96, 192 // factor)
#         self.up1 = Up(192, 128 // factor, bilinear)
#         self.up2 = Up(128, 64 // factor, bilinear)
#         self.up3 = Up(64, 32 // factor, bilinear)
#         self.up4 = Up(32, 16, bilinear)
#         self.outc = OutConv(16, n_classes)

class UNet(nn.Module): # 
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        model_scale = 0
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 56)
        self.down3 = Down(56, 96)
        factor = 2 if bilinear else 1
        self.down4 = Down(96, 192 // factor)
        self.up1 = Up(192, 112 // factor, bilinear)
        self.up2 = Up(112, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print("x1:",x1.size())
        # print("x2:",x2.size())
        # print("x3:",x3.size())
        # print("x4:",x4.size())
        # print("x5:",x5.size())

        x = self.up1(x5, x4)
        # print("up1(x5, x4):",x.size())
        x = self.up2(x, x3)
        # print("up2(x, x3):",x.size())
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        #logits = torch.sigmoid(logits)
        return logits
if __name__ == '__main__':
    device = torch.device("cpu")

    model= UNet(n_channels=3, n_classes=2, bilinear=False)

    batch_size = 1  #批处理大小
    input_shape = (3, 240,320)   #输入数据,改成自己的输入shape

    # #set the model to inference mode
    model.eval()

    x = torch.randn(batch_size, *input_shape)   # 生成张量
    x = x.to(device)
    out_ = model(x)
    print("out =:  " ,out_.size())
