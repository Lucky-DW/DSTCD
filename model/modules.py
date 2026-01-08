import torch
import torch.nn as nn
import torch.nn.functional as F

class Self_MultiheadAttention(nn.Module):
    def __init__(self, dropout=0.2, d_model=64, n_head=4):
        super(Self_MultiheadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, input):

        attn_output, attn_weight = self.attention(input, input, input)  # (Q,K,V)

        output = input + self.dropout1(attn_output)

        output = self.norm1(output)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
        output = output + self.dropout3(ff_output)
        output = self.norm2(output)

        return output

class CrossTransformer(nn.Module):
    def __init__(self, dropout, d_model=512, n_head=4):
        super(CrossTransformer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, input1, input2):
        # dif_as_kv
        dif = input2 - input1
        output_1 = self.cross(input1, dif)  # (Q,K,V)
        output_2 = self.cross(input2, dif)  # (Q,K,V)

        return output_1, output_2
    def cross(self, input,dif):
        # RSICCformer_D (diff_as_kv)
        attn_output, attn_weight = self.attention(input, dif, dif)  # (Q,K,V)

        output = input + self.dropout1(attn_output)

        output = self.norm1(output)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
        output = output + self.dropout3(ff_output)
        output = self.norm2(output)
        return output

class PyramidPool(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size):
        super(PyramidPool, self).__init__()
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.BatchNorm2d(out_channels, momentum=0.95),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape
        output = F.upsample_bilinear(self.features(x), size[2:])
        return output

class SP_fusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SP_fusion, self).__init__()

        assert out_channels % 4 == 0
        conv_out = int(out_channels / 4)

        self.init = nn.Conv2d(in_channels, out_channels,kernel_size=1)

        self.layer5a = PyramidPool(out_channels, conv_out, 1)
        self.layer5b = PyramidPool(out_channels, conv_out, 2)
        self.layer5c = PyramidPool(out_channels, conv_out, 3)
        self.layer5d = PyramidPool(out_channels, conv_out, 6)

        self.PF = nn.Conv2d(2 * out_channels,out_channels,kernel_size=1)

        self.bn=nn.BatchNorm2d(out_channels,eps=0.001)

    def forward(self, x):
        x = self.init(x)

        x = torch.cat([
            x,
            self.layer5a(x),
            self.layer5b(x),
            self.layer5c(x),
            self.layer5d(x),
        ], 1)

        x = self.bn(self.PF(x))

        return F.relu(x,inplace=True)

class OneConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ThreeConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class FourConv(nn.Module):
        """(convolution => [BN] => ReLU) * 2"""

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        def forward(self, x):
            return self.double_conv(x)

class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception,self).__init__()
        self.Branch1x1=nn.Conv2d(in_channels,out_channels//4,kernel_size=1)


        self.Branch3x3_1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1)
        self.Branch3x3=nn.Conv2d(out_channels//4,out_channels//4,kernel_size=3,padding=1)


        self.Branch5x5_1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1)
        self.Branch5x5=nn.Conv2d(out_channels//4,out_channels//4,kernel_size=5,padding=2)

        self.Branchmax1x1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1)

        self.bn=nn.BatchNorm2d(out_channels,eps=0.001)


    def forward(self, x):
        branch1x1=self.Branch1x1(x)

        branch2_1=self.Branch3x3_1(x)
        branch2_2=self.Branch3x3(branch2_1)

        branch3_1=self.Branch5x5_1(x)
        branch3_2=self.Branch5x5(branch3_1)

        branchpool4_1=F.max_pool2d(x,kernel_size=3,stride=1,padding=1)
        branchpool4_2=self.Branchmax1x1(branchpool4_1)

        outputs=[branch1x1,branch2_2,branch3_2,branchpool4_2]
        x= torch.cat(outputs,1)
        x=self.bn(x)
        return F.relu(x,inplace=True)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        diffx_2 = torch.div(diffX, 2, rounding_mode='floor')
        diffy_2 = torch.div(diffY, 2, rounding_mode='floor')

        x1 = F.pad(x1, [diffx_2, diffX - diffx_2,
                        diffy_2, diffY - diffy_2])

        x = torch.cat([x2, x1], dim=1)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("cam输入尺寸大小：",x.size())
        # print("最大池化后大小：",self.avg_pool(x).size())
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        # print("经过cam输出通道特征尺寸:",self.sigmoid(avgout + maxout).size())
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out