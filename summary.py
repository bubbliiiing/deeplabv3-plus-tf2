#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from nets.deeplab import Deeplabv3

if __name__ == "__main__":
    model = Deeplabv3([512,512,3], 21, backbone='mobilenet')
    model.summary()

    for i,layer in enumerate(model.layers):
        print(i,layer.name)
