import numpy as np

from option import get_option
from model.Encoder_p2t import Encoder
from model.Res2Net_v1b import res2net50_v1b_26w_4s
from model.Transformer import Transformer
from model.ERM import ERM
from model.FEM import FEM
import torch
import torch.nn as nn


# 通道减少 -> 通道注意力
class CCM(nn.Module):
    def __init__(self, infeature, out, redio):
        super(CCM, self).__init__()
        self.down = nn.Conv2d(infeature, out, kernel_size=1, stride=1)
        self.channel_attention = ChannelAttention(out, redio)

    def forward(self, x):
        x = self.down(x)
        w = self.channel_attention(x)
        return x * w


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 定义全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 定义全局最大池化
        # 定义CBAM中的通道依赖关系学习层，注意这里是使用1x1的卷积实现的，而不是全连接层
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))  # 实现全局平均池化
        max_out = self.fc(self.max_pool(x))  # 实现全局最大池化
        out = avg_out + max_out  # 两种信息融合
        # 最后利用sigmoid进行赋权
        return self.sigmoid(out)


class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True, pretrain=opt.res2net_path)
        self.encoder = Encoder(opt)
        # global context
        self.encoder_tf_ss = Transformer(depth=2,
                                         num_heads=1,
                                         embed_dim=256,
                                         mlp_ratio=3,
                                         num_patches=196)

        self.encoder_shaper_7 = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 1024), nn.GELU())
        self.encoder_shaper_14 = nn.Sequential(nn.LayerNorm(320), nn.Linear(320, 256), nn.GELU())
        self.encoder_shaper_28 = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU())
        self.encoder_shaper_56 = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 16), nn.GELU())

        self.encoder_merge7_14 = nn.Sequential(nn.BatchNorm2d(512),
                                               nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=True),
                                               nn.LeakyReLU())
        self.encoder_merge28_14 = nn.Sequential(nn.BatchNorm2d(512),
                                                nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=True),
                                                nn.LeakyReLU())
        self.encoder_merge56_14 = nn.Sequential(nn.BatchNorm2d(512),
                                                nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=True),
                                                nn.LeakyReLU())

        self.encoder_pred = nn.Sequential(nn.LayerNorm(256),
                                          nn.Linear(256, 256),
                                          nn.GELU(),
                                          nn.LayerNorm(256),
                                          nn.Linear(256, 256),
                                          nn.GELU(),
                                          nn.LayerNorm(256),
                                          nn.Linear(256, 1)
                                          )
        # main network
        self.transformer = nn.ModuleList([Transformer(depth=d,
                                                      num_heads=n,
                                                      embed_dim=e,
                                                      mlp_ratio=m,
                                                      num_patches=p) for d, n, e, m, p in opt.transformer])

        self.FEM7_14 = FEM(emb_dim=320, hw=7, cur_stg=512)
        self.FEM14_28 = FEM(emb_dim=128, hw=14, cur_stg=320)
        self.FEM28_56 = FEM(emb_dim=64, hw=28, cur_stg=128)

        self.ERM_7 = ERM(inc=512, outc=1024, hw=7, embed_dim=512, num_patches=49)
        self.ERM_14 = ERM(inc=320, outc=256, hw=14, embed_dim=320, num_patches=196)
        self.ERM_28 = ERM(inc=128, outc=64, hw=28, embed_dim=128, num_patches=784)
        self.ERM_56 = ERM(inc=64, outc=16, hw=56, embed_dim=64, num_patches=3136)

        # CNN特征通道压缩
        self.down1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1)
        self.down2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1)
        self.down3 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        self.down4 = nn.Conv2d(256, 128, kernel_size=1, stride=1)

        # CCM模块压缩通道的通道注意力  CNN
        self.ccm_c1 = CCM(1024, 512, redio=16)
        self.ccm_c2 = CCM(512, 128, redio=8)
        self.ccm_c3 = CCM(256, 80, redio=4)
        self.ccm_c4 = CCM(128, 32, redio=4)

    def forward(self, x):
        B = x.shape[0]
        cnn_list = self.resnet(x)

        # P2T encoder
        out_7r, out_14r, out_28r, out_56r = self.encoder(x)  # is a cat_feature, list in shape of 16, 32, 64, 128

        pred = list()

        out_7r = out_7r.flatten(2).transpose(1, 2)
        out_14r = out_14r.flatten(2).transpose(1, 2)
        out_28r = out_28r.flatten(2).transpose(1, 2)
        out_56r = out_56r.flatten(2).transpose(1, 2)

        # 经过Transformer
        out_7, out_14, out_28, out_56 = [tf(o, peb) for tf, o, peb in zip(self.transformer,
                                                                          [out_7r, out_14r, out_28r, out_56r],
                                                                          [False, False, False,
                                                                           False])]  # B, patch, feature

        # CNN特征
        c4 = self.down4(cnn_list[0])
        c3 = self.down3(cnn_list[1])
        c2 = self.down2(cnn_list[2])
        c1 = self.down1(cnn_list[3])

        # 经过通道压缩 通道注意力
        c4 = self.ccm_c4(c4)  # 56 56 32
        c3 = self.ccm_c3(c3)  # 28 28 80
        c2 = self.ccm_c2(c2)  # 14 14 128
        c1 = self.ccm_c1(c1)  # 7 7 512

        # 7  粗预测图
        p1_7, p2_7, out_7 = self.ERM_7(out_7)
        pred.append(p1_7)
        pred.append(p2_7)

        # 14 14 128
        out_14 = self.FEM7_14(out_14, out_7, c2)
        p1_14, p2_14, out_14 = self.ERM_14(out_14)
        pred.append(p1_14)
        pred.append(p2_14)

        # 28 28 80
        out_28 = self.FEM14_28(out_28, out_14, c3)
        p1_28, p2_28, out_28 = self.ERM_28(out_28)
        pred.append(p1_28)
        pred.append(p2_28)

        # 56 56 32
        out_56 = self.FEM28_56(out_56, out_28, c4)
        p1_56, p2_56, out_56 = self.ERM_56(out_56)
        pred.append(p1_56)
        pred.append(p2_56)

        return pred


if __name__ == '__main__':
    a = np.random.random((2, 3, 224, 224))
    b = np.random.random((1, 3, 224, 224))
    c = torch.Tensor(a).cuda()
    d = torch.Tensor(b).cuda()
    opt = get_option()
    DSRNet = Net(opt).cuda()

    DSRNet(c)
