from torch import nn
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
from ..module.activation import activations 


# # from https://codeload.github.com/houqb/ssdlite-pytorch-mobilenext/zip/refs/heads/master
"""
torch.Size([1, 32, 112, 112]) 0
torch.Size([1, 96, 56, 56]) 1
torch.Size([1, 144, 56, 56]) 2
torch.Size([1, 192, 28, 28]) 3
torch.Size([1, 192, 28, 28]) 4
torch.Size([1, 192, 28, 28]) 5
torch.Size([1, 288, 14, 14]) 6
torch.Size([1, 288, 14, 14]) 7
torch.Size([1, 288, 14, 14]) 8
torch.Size([1, 384, 7, 7]) 9
torch.Size([1, 384, 7, 7]) 10
torch.Size([1, 384, 7, 7]) 11
torch.Size([1, 384, 7, 7]) 12
torch.Size([1, 576, 4, 4]) 13
torch.Size([1, 576, 4, 4]) 14
torch.Size([1, 576, 4, 4]) 15
torch.Size([1, 576, 4, 4]) 16
torch.Size([1, 960, 4, 4]) 17
torch.Size([1, 960, 4, 4]) 18
torch.Size([1, 960, 4, 4]) 19
torch.Size([1, 1280, 4, 4]) 20
2.25864

"""
class DepthWiseAugmentation(nn.Module):
    """
    This is the kernel refine module
    """
    def __init__(self, in_dim=1):
        super(DepthWiseAugmentation, self).__init__()
        self.bn = nn.BatchNorm2d(1)
        #self.bn2 = nn.BatchNorm2d(in_dim)

    def forward(self,x,m):
        out = m(x)
        [C_out, C_in, kernel_size, kernel_size] = m.weight.shape
        w = m.weight.sum(2).sum(2)
        w = w[:, :, None, None]
        w = w.permute(1,0,2,3)
        out_diff = F.conv2d(input=x, weight=w, stride=1, padding=0)

        # outs = self.att(out_normal, out_normal - theta * out_diff)
        outs = F.sigmoid(self.bn(out_diff))*out
        return outs


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1,activation=nn.ReLU6):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            activation(inplace=True),#nn.ReLU6(inplace=True)
        )


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride,activation=nn.ReLU6):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        activation(inplace=True),#nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup,activation=nn.ReLU6):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        activation(inplace=True),#nn.ReLU6(inplace=True)
    )


class SGBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, keep_3x3=False,activation=nn.ReLU6):
        super(SGBlock, self).__init__()
        assert stride in [1, 2]

        hidden_dim = inp // expand_ratio
        if hidden_dim < oup / 6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)  # + 16

        # self.relu = nn.ReLU6(inplace=True)
        self.identity = False
        self.identity_div = 1
        self.expand_ratio = expand_ratio
        self.aug =None
        if expand_ratio == 2:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                activation(inplace=True), #nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                activation(inplace=True), #nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
        elif inp != oup and stride == 1 and keep_3x3 == False:
            self.conv = nn.Sequential(
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                activation(inplace=True),#nn.ReLU6(inplace=True),
            )
        elif inp != oup and stride == 2 and keep_3x3 == False:
            self.conv = nn.Sequential(
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                activation(inplace=True),#nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            if keep_3x3 == False:
                self.identity = True
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                activation(inplace=True), #nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                activation(inplace=True), #nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, 1, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
            self.aug = DepthWiseAugmentation(inp)

    def forward(self, x):
        if self.aug is None:
            out = self.conv(x)
        else:
            out = x
            #for i,m in enumerate(self.conv):
            #    if i==0:
            out=self.aug(out,self.conv[0])
        
            out=self.conv[1:](out)
            


        if self.identity:
            return out + x
        else:
            return out


def sg_block_extra(inp, oup, stride, expand_ratio, last=False,activation=None):
    hidden_dim = int(inp * expand_ratio)
    conv = nn.Sequential(
        # dw
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        activation(inplace=True), #nn.ReLU6(inplace=True),
        # pw
        nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
        nn.BatchNorm2d(hidden_dim),
        # pw
        nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        activation(inplace=True), #nn.ReLU6(inplace=True),
    )
    return conv


class MobileNeXt(nn.Module):
    def __init__(self,pretrained=True,out_stages=[], activation=None,num_classes=1000, width_mult=1.):
        super(MobileNeXt, self).__init__()

        # setting of inverted residual blocks
        if out_stages==[5,8,12]:  # the small version, ouput channels: [192,288,384] 
            self.cfgs = [
            # t, c, n, s
            [2, 96, 1, 2],
            [6, 144, 1, 1],
            [6, 192, 3, 2],
            [6, 288, 3, 2],
            [6, 384, 4, 2],
            #[6, 576, 4, 2],
            #[6, 960, 3, 1],
            # [6, 1280, 1, 1],
            ]
        elif out_stages==[5,12,16]:  # the medium version , ouput channels: [192,384,576] 
            self.cfgs = [
            # t, c, n, s
            [2, 96, 1, 2],
            [6, 144, 1, 1],
            [6, 192, 3, 2],
            [6, 288, 3, 2],
            [6, 384, 4, 1],
            [6, 576, 4, 2],
            #[6, 960, 3, 1],
            # [6, 1280, 1, 1],
            ]
        elif out_stages==[5,12,19]:  # # the medium version , ouput channels: [192,384,960]
            self.cfgs = [
            # t, c, n, s
            [2, 96, 1, 2],
            [6, 144, 1, 1],
            [6, 192, 3, 2],
            [6, 288, 3, 2],
            [6, 384, 4, 1],
            [6, 576, 4, 2],
            [6, 960, 3, 1],
            # [6, 1280, 1, 1],
            ]
        else:
            raise ValueError('Not supoort the outstages of mobilenext!')

        act_layer = activations[activation]
        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2, activation=act_layer)]
        # building inverted residual blocks
        block = SGBlock
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            if c == 1280 and width_mult < 1:
                output_channel = 1280
            layers.append(block(input_channel, output_channel, s, t, n == 1 and s == 1,activation=act_layer))
            input_channel = output_channel
            for i in range(n - 1):
                layers.append(block(input_channel, output_channel, 1, t, activation=act_layer))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        input_channel = output_channel
        output_channel = _make_divisible(input_channel, 4)
        self.out_stages = out_stages
        '''self.extras = nn.ModuleList([
            sg_block_extra(1280, 512, 2, 0.2),
            sg_block_extra(512, 256, 2, 0.25),
            sg_block_extra(256, 256, 2, 0.5),
            sg_block_extra(256, 128, 2, 0.5)
        ])'''

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):

        features = []
        '''for i,m in enumerate(self.features):
            x = m(x)
            if i in self.out_stages:
                features.append(x)'''
        pre = -1
        for i in range(len(self.out_stages)):
            x = self.features[pre+1:self.out_stages[i]+1](x)
            pre = self.out_stages[i]
            features.append(x)
        '''for i in range(13):
            x = self.features[i](x)

        for i in range(5):
            x = self.features[13].conv[i](x)
        features.append(x)
        for i in range(5, len(self.features[13].conv)):
            x = self.features[13].conv[i](x)

        for i in range(14, len(self.features) - 1):
            x = self.features[i](x)

        for i in range(8):
            x = self.features[20].conv[i](x)
        features.append(x)
        for i in range(8, len(self.features[20].conv)):
            x = self.features[20].conv[i](x)

        for i in range(len(self.extras)):
            x = self.extras[i](x)
            features.append(x)'''
        return tuple(features)

def mobilenext(**kwargs):
    model = MobileNeXt(**kwargs)
    if kwargs['pretrained']:
        print('successfully load!')
        #pretrained_state_dict= torch.hub.load_state_dict_from_url('blob:https://github.com/392f6b46-e2ed-4cf2-aa85-890b5b89e949',progress=True)
        model.load_state_dict( torch.load('weights/mnext.pth.tar'), strict=False)
    return model

if __name__ == '__main__':
    net = mobilenext()
    x = torch.rand([1,3,224,224])
    net.eval()
    net(x)
    print(sum([torch.numel(i) for i in net.parameters()])/10**6)