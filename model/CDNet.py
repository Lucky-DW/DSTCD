import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules import DoubleConv, OneConv, OutConv, Down, Up
from model.src.ASpanFormer.aspanformer import ASpanFormer
from model.src.config.default import get_cfg_defaults
from model.src.utils.misc import lower_config

class opt_sar_CDNet(nn.Module):
    def __init__(self, n_channels=6, n_classes=1, channel_li=[64, 128, 256, 512],
                head=4, dropout=0.2, Transfer_channel=[128, 256],
                Transfer=True, Structure=True, Sementic=True):
        super(opt_sar_CDNet, self).__init__()
        self.Transfer = Transfer
        self.n_head = head
        self.dropout = dropout
        self.channels = channel_li
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.Transfer_C = Transfer_channel
        self.structure = Structure
        self.sementic = Sementic

        self.init = nn.Conv2d(self.n_channels, self.channels[0], kernel_size=7, padding=3, stride=2, bias=False)
        self.down = Down()
        self.up = Up()
        self.ecoder_block1 = DoubleConv(self.channels[0], self.channels[1])
        self.ecoder_block2 = DoubleConv(self.channels[1], self.channels[2])
        self.ecoder_block3 = DoubleConv(self.channels[2], self.channels[3])
        self.ecoder_block4 = DoubleConv(self.channels[3], self.channels[3])

        self.f1 = OneConv(self.channels[3], self.channels[3])
        self.f2 = OneConv(2 * self.channels[3], self.channels[2])
        self.f3 = OneConv(2 * self.channels[2], self.channels[1])
        self.f4 = OneConv(2 * self.channels[1], self.channels[0])

        self.outc = OutConv(self.channels[0], self.n_classes)

        if self.Transfer:
            ##########################################structure#####################################################
            if self.structure:
                assert self.channels[1] % 4 == 0
                self.D_conv = nn.Conv2d(self.channels[1], self.channels[1] // 4, kernel_size= 1)
                self.split1 = OneConv(self.channels[1] // 4, self.channels[1] // 4)
                self.split2 = OneConv(self.channels[1] // 4, self.channels[1] // 4)
                self.split3 = OneConv(self.channels[1] // 4, self.channels[1] // 4)
                self.split4 = OneConv(self.channels[1] // 4, self.channels[1] // 4)

                self.D_D = OneConv(2 * self.channels[1], self.channels[1])
            else:
                self.low2 = DoubleConv(2 * self.Transfer_C[0], self.Transfer_C[0])
                self.low2f = OneConv(2 * self.channels[1], self.channels[1])
            ###########################################Sementic######################################################
            if self.sementic:
                self.F2TQ_Liner = nn.Linear(self.channels[-1], self.Transfer_C[-1])
                self.T1K_Liner = nn.Linear(self.Transfer_C[-1], self.Transfer_C[-1])
                self.T1V_Liner = nn.Linear(self.Transfer_C[-1], self.Transfer_C[-1])
                self.T2K_Liner = nn.Linear(self.Transfer_C[-1], self.Transfer_C[-1])
                self.T2V_Liner = nn.Linear(self.Transfer_C[-1], self.Transfer_C[-1])
                self.FTQ_Liner = nn.Linear(self.channels[-1], self.channels[-1])
                self.TFK_Liner = nn.Linear(self.channels[-1], self.channels[-1])
                self.TFV_Liner = nn.Linear(self.channels[-1], self.channels[-1])
                self.T1_MHA = nn.MultiheadAttention(self.Transfer_C[-1], self.n_head, dropout=dropout)
                self.T2_MHA = nn.MultiheadAttention(self.Transfer_C[-1], self.n_head, dropout=dropout)
                self.T_MHA = nn.MultiheadAttention(self.channels[-1], self.n_head, dropout=dropout)
            else:
                self.deep2 = DoubleConv(2 * self.Transfer_C[-1], 2 * self.Transfer_C[-1])
                self.deep2f = OneConv(2 * self.channels[-1], self.channels[-1])

    def forward(self,t1, t2):
        x = torch.cat((t1, t2), dim=1)
        features = self.Simple_encoder(x)
        output_1  = self.Simple_decoder(features)
        output_1 = F.interpolate(output_1, size=x.size()[2:], mode='bilinear', align_corners=True)
        output_1 = torch.sigmoid(self.outc(output_1))

        if self.Transfer:
            x1 = t1[:, :-1, :, :]
            x2 = t2[:, :-1, :, :]
            transfer_out_low, transfer_out_deep = self.Transfer_knowlege_Net(x1, x2)
            t1_low, t2_low = transfer_out_low[0], transfer_out_low[1]
            t1_deep, t2_deep = transfer_out_deep[0], transfer_out_deep[1]

            ##########################################structure#####################################################
            if self.structure:
                D1 = self.T_D_fusion(t1_low, features[0])
                D2 = self.T_D_fusion(t2_low, features[0])
                features[0] = self.D_D(torch.cat((D1, D2), dim=1))
            else:
                low = self.low2(torch.cat((t1_low, t2_low), dim=1))
                features[0] = self.low2f(torch.cat((features[0], low), dim=1))
            ########################################################################################################

            ###########################################sementic######################################################
            if self.sementic:
                D_B, D_C, D_H, D_W = t1_deep.shape[0], t1_deep.shape[1], t1_deep.shape[2], t1_deep.shape[3]
                F_B, F_C, F_H, F_W = features[-1].shape[0],features[-1].shape[1],features[-1].shape[2],features[-1].shape[3]
                D_Q = self.F2TQ_Liner(features[-1].reshape(F_B, F_C, -1).permute(0, 2, 1))
                t1_K = self.T1K_Liner(t1_deep.reshape(D_B, D_C, -1).permute(0, 2, 1))
                t1_V = self.T1V_Liner(t1_deep.reshape(D_B, D_C, -1).permute(0, 2, 1))
                t2_K = self.T2K_Liner(t2_deep.reshape(D_B, D_C, -1).permute(0, 2, 1))
                t2_V = self.T2V_Liner(t2_deep.reshape(D_B, D_C, -1).permute(0, 2, 1))
                T1_output, T1_weight = self.T1_MHA(D_Q, t1_K, t1_V)
                T2_output, T2_weight = self.T2_MHA(D_Q, t2_K, t2_V)
                T_output = torch.cat((T1_output, T2_output), dim=-1)

                q = self.FTQ_Liner(features[-1].reshape(F_B, F_C, -1).permute(0, 2, 1))
                k = self.TFK_Liner(T_output)
                v = self.TFK_Liner(T_output)
                output, weight = self.T_MHA(q, k, v)
                features[-1] = features[-1] + output.permute(0, 2, 1).reshape(F_B, F_C, F_H, F_W)
            else:
                deep = self.deep2(torch.cat((t1_deep, t2_deep), dim=1))
                features[-1] = self.deep2f(torch.cat((features[-1], deep), dim=1))
            #########################################################################################################

        output_2  = self.Simple_decoder(features)
        output_2 = F.interpolate(output_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        output_2 = torch.sigmoid(self.outc(output_2))

        return output_1, output_2

    def Simple_decoder(self, features):
        feat_4 = self.f1(features[-1])
        feat_3 = self.f2(torch.cat((feat_4, features[-2]),dim=1))
        feat_2 = self.f3(self.up(feat_3, features[1]))
        feat_1 = self.f4(self.up(feat_2, features[0]))

        return feat_1

    def Transfer_knowlege_Net(self,t1, t2):
        batchsize = t1.size(0)
        t1 = t1[:,0,...].unsqueeze(1)
        t2 = t2[:,0,...].unsqueeze(1)

        config = get_cfg_defaults()
        config.merge_from_file('model/aspan_test.py')
        _config = lower_config(config)
        matcher = ASpanFormer(config=_config['aspan'])

        state_dict = torch.load('model/ck/outdoor.ckpt')['state_dict']
        #state_dict = torch.load("weights/outdoor.ckpt",map_location='cpu')['state_dict']
        matcher.load_state_dict(state_dict, strict=False)
        matcher.to(t1.device)
        # matcher.cuda()  # matcher.cpu()
        matcher.eval()

        layer4_feat, layer1_feat = matcher.backbone(torch.cat([t1, t2], dim=0))

        layer1_feat_t1, layer1_feat_t2 = layer1_feat.split(batchsize)     #(B, 128, H/2, W/2)
        layer4_feat_t1, layer4_feat_t2 = layer4_feat.split(batchsize)     #(B, 256, H/8, W/8)

        return [layer1_feat_t1, layer1_feat_t2],[layer4_feat_t1, layer4_feat_t2]

    def Simple_encoder(self, x):
        x1 = self.init(x)
        x1 = self.ecoder_block1(x1)
        x2 = self.down(x1)
        x2 = self.ecoder_block2(x2)
        x3 = self.down(x2)
        x3 = self.ecoder_block3(x3)
        x4 = self.ecoder_block4(x3)

        return [x1, x2, x3, x4]

    def calculate_similarity_map(self, T, D):
        B, C, H, W = T.shape
        assert D.shape == (B, C, H, W)

        T_flat = T.permute(0, 2, 3, 1).reshape(B, H * W, C)
        D_flat = D.permute(0, 2, 3, 1).reshape(B, H * W, C)

        similarity = F.cosine_similarity(T_flat, D_flat, dim=-1)

        similarity_map = similarity.view(B, 1, H, W)

        return similarity_map

    def T_D_fusion(self, T, D):

        B, C, H, W = T.shape
        assert C % 4 == 0 #通道数 C 必须是 4 的倍数

        D_Sim = self.D_conv(D)

        x1, x2, x3, x4 = torch.split(T, C // 4, dim=1)

        x1 = self.calculate_similarity_map(self.split1(x1), D_Sim) * D_Sim
        x2 = self.calculate_similarity_map(self.split2(x2), D_Sim) * D_Sim
        x3 = self.calculate_similarity_map(self.split3(x3), D_Sim) * D_Sim
        x4 = self.calculate_similarity_map(self.split4(x4), D_Sim) * D_Sim

        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x + D

if __name__ == '__main__':
    in1=torch.randn(8, 3, 256, 256)
    in2=torch.randn(8, 3, 256, 256)
    net=opt_sar_CDNet()
    net.train()
    out1, out2 =net(in1,in2)
    print(out1.shape)
    print(out2.shape)