import torch.nn as nn
import torch
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        # self.relu = nn.LeakyReLU(0.2, True) if relu else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def pixel_unshuffle(input, upscale_factor):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.
    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet
    Date:
        01/Jan/2019
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)

class PixelUnShuffle(nn.Module):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.
    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet
    Date:
        01/Jan/2019
    """

    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            BasicConv(channel, channel // reduction, 1, stride=1, padding=(1 - 1) // 2, relu=False, bias=True),

            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.2, True),
            BasicConv(channel // reduction, channel, 1, stride=1, padding=(1 - 1) // 2, relu=False, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ChannelPool(nn.Module):
    def forward(self, x):
        # return (torch.mean(x, 1).unsqueeze(1))
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


##########################################################################


## Dual Attention Block (DAB)
class DAB(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=16,
            bias=True, bn=False, act=nn.ReLU(inplace=True)):

        super(DAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(
                BasicConv(n_feat, n_feat, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False, bias=True),
            )
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)

        self.SA = spatial_attn_layer()  ## Spatial Attention
        self.CA = CALayer(n_feat, reduction)  ## Channel Attention
        # self.body = nn.Sequential(*modules_body)
        self.conv1x1 = BasicConv(n_feat * 2, n_feat, 1, stride=1, padding=(1 - 1) // 2, relu=False, bias=True)

    def forward(self, x):
        # res = self.body(x)
        res = x
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        return res

class NoiseMapEn_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(NoiseMapEn_module, self).__init__()

        self.activaton = nn.Sigmoid()
        # self.e_lambda = e_lambda
        # self.a = nn.Parameter(torch.ones(1)*1)
        # self.b = nn.Parameter(torch.ones(1)*0.5*1e-8)
        # self.c = nn.Parameter(torch.ones(1)*1e-8)

    def forward(self, x):
        # b, c, h, w = x.size()
        #
        # n = w * h

        var = (x - x.mean(3,keepdim=True).mean(2,keepdim=True)).pow(2)
        # all_ave_var = var.mean(dim=[1, 2, 3], keepdim=True)
        spa_ave_var = var.mean(3,keepdim=True).mean(2,keepdim=True)
        cha_ave_var = var.mean(1, keepdim=True)
        # y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        # zz=x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n
        # y = (10*var)/(ave_var+1e-12)
        y_spa = (10 * var) / (spa_ave_var + 1e-16)
        y_cha = (10 * var) / (cha_ave_var + 1e-16)
        # y=y_all*y_layer
        weight_spa = self.activaton(y_spa)
        weight_cha = self.activaton(y_cha)
        weight = weight_spa * weight_cha
        return x * weight

class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        # zz=x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n
        return x * self.activaton(y)

class Corrector(nn.Module):
    def __init__(self, nf = 64, nf_2=64, input_para=1,num_blocks= 5):
        super(Corrector, self).__init__()
        self.head_noisy = BasicConv(1*4, nf_2, 3, stride=1, padding=(3 - 1) // 2, relu=False)
        self.head_img_Fea = BasicConv(64, nf_2, 3, stride=1, padding=(3 - 1) // 2, relu=False)
        self.m_down = PixelUnShuffle(upscale_factor=2)
        self.m_up = nn.PixelShuffle(2)

        self.ConvNet_Input = nn.Sequential(*[
            BasicConv(nf*2, nf, 1, stride=1, padding=(1 - 1) // 2, relu=False),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True)
        ])
        self.ConvNet_f0noise = nn.Sequential(*[
            BasicConv(nf*2, nf, 1, stride=1, padding=(1 - 1) // 2, relu=False),
        ])
        # self.att = DAB(nf)
        self.att = NoiseMapEn_module()
        self.conv1 = BasicConv(nf, input_para*4, 3, stride=1, padding=(3 - 1) // 2, relu=False)

    def forward(self, noisy_map, input_img_feature, feature_maps):
        input_img_down = self.head_img_Fea(input_img_feature)
        para_maps = self.head_noisy(self.m_down(noisy_map))
        cat_f0noise = self.ConvNet_f0noise(torch.cat((input_img_down, feature_maps), dim=1))
        cat_input = self.ConvNet_Input(torch.cat((para_maps, cat_f0noise), dim=1))
        return self.m_up(self.conv1(self.att(cat_input)))



class Predictor(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(Predictor, self).__init__()

        self.ConvNet = nn.Sequential(*[
            BasicConv(in_nc*4, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True)
        ])
        self.m_down = PixelUnShuffle(upscale_factor=2)
        self.m_up = nn.PixelShuffle(2)
        # self.att = DAB(nf)
        # self.att = simam_module()
        self.att = NoiseMapEn_module()
        self.conv1 = BasicConv(nf, 1*4, 3, stride=1, padding=(3 - 1) // 2, relu=False)

    def forward(self, input):
        input_down = self.m_down(input)
        att = self.att(self.ConvNet(input_down))
        return self.m_up(self.conv1(att)), att


class SidePool(nn.Module):
    def forward(self, x,a):
        return torch.cat((torch.mean(x, a).unsqueeze(a), torch.max(x, a)[0].unsqueeze(a)), dim=a)


class ChannelPool_2D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelPool_2D, self).__init__()
        self.Conv = BasicConv(channel, 2, 1, stride=1, padding=(1 - 1) // 2, relu=False)

    def forward(self, x):
        return torch.cat((torch.mean(x, 1).unsqueeze(1),
                          torch.max(x, 1)[0].unsqueeze(1),
                          self.Conv(x)), dim=1)

class SA_attention(nn.Module):
    def __init__(self, channel):
        super(SA_attention, self).__init__()
        self.compress = ChannelPool_2D(channel)
        self.conv_du = nn.Sequential(
            BasicConv(4, 1, 7, stride=1, padding=(7 - 1) // 2, relu=False, bias=True),
            nn.ReLU(inplace=True),
            BasicConv(1, 1, 7, stride=1, padding=(7 - 1) // 2, relu=False, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.compress(x)
        y = self.conv_du(y)
        return x * y

class ChannelPool_1D(nn.Module):
    def forward(self, x):
        return torch.cat((x.mean(3).mean(2,keepdim=True), x.max(3)[0].max(2,keepdim=True)[0]), 2)

class CA_attention(nn.Module):
    def __init__(self, channel):
        super(CA_attention, self).__init__()
        self.compress = ChannelPool_1D()
        self.soft = nn.Softmax(dim=2)
        self.conv_b = nn.Sequential(
            BasicConv(channel, 2, 1, stride=1, padding=(1 - 1) // 2, relu=False),
            BasicConv(2, 2, 7, stride=1, padding=(7 - 1) // 2, relu=True),
            BasicConv(2, 2, 7, stride=1, padding=(7 - 1) // 2, relu=False)
        )
        self.cat = nn.Sequential(
            BasicConv(channel, 4, 1, stride=1, padding=(1 - 1) // 2, relu=True),
            BasicConv(4, channel, 1, stride=1, padding=(1 - 1) // 2, relu=False)
        )
        self.conv_du = nn.Sequential(
            BasicConv(4, 1, 1, stride=1, padding=(1 - 1) // 2, relu=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        a = self.compress(x)
        b = self.soft(self.conv_b(x).reshape(n, 2, h*w))
        x_T = x.reshape(n, c, h*w).transpose(1, 2)
        b = torch.matmul(b, x_T).transpose(1, 2)

        y = self.cat(torch.cat((a, b), 2).unsqueeze(3))
        y = self.conv_du(y.transpose(1, 2)).transpose(1, 2)
        return x * y

class inplaceCA(nn.Module):
    def __init__(self, channel):
        super(inplaceCA, self).__init__()
        self.Conv = BasicConv(4, 4, 1, stride=1, padding=(1 - 1) // 2, relu=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(3,keepdim=True).mean(2,keepdim=True)
        y = self.Conv(y)
        return x + x*self.sig(y)

class SA_attention0(nn.Module):
    def __init__(self, channel):
        super(SA_attention0, self).__init__()
        self.compress = ChannelPool_2D(channel)
        self.inplaceCA = inplaceCA(channel)
        self.conv_du = nn.Sequential(
            BasicConv(4, 4, 3, stride=1, padding=(3 - 1) // 2, relu=False, bias=True),
            self.inplaceCA,
            BasicConv(4, 1, 1, stride=1, padding=(1 - 1) // 2, relu=False, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.compress(x)
        y = self.conv_du(y)
        return x * y

class inplaceSA(nn.Module):
    def __init__(self, channel):
        super(inplaceSA, self).__init__()
        self.Conv = BasicConv(channel, 1, 3, stride=1, padding=(3 - 1) // 2, relu=False)
        self.sig=nn.Sigmoid()

    def forward(self, x):
        y = self.Conv(x)
        x = x * self.sig(y)
        return torch.cat((x.mean(3).mean(2,keepdim=True), x.max(3)[0].max(2,keepdim=True)[0]), 2)

class CA_attention0(nn.Module):
    def __init__(self, channel):
        super(CA_attention0, self).__init__()
        self.compress = ChannelPool_1D()
        self.inplaceSA=inplaceSA(channel)
        self.cat = nn.Sequential(
            BasicConv(channel, 4, 1, stride=1, padding=(1 - 1) // 2, relu=True),
            BasicConv(4, channel, 1, stride=1, padding=(1 - 1) // 2, relu=False)
        )
        self.conv_du = nn.Sequential(
            BasicConv(4, 1, 1, stride=1, padding=(1 - 1) // 2, relu=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        a = self.compress(x)
        b = self.inplaceSA(x)

        y = self.cat(torch.cat((a, b), 2).unsqueeze(3))
        y = self.conv_du(y.transpose(1, 2)).transpose(1, 2)
        return x * y



class attention(nn.Module):
    def __init__(self, channel, i, reduction=16):
        super(attention, self).__init__()
        self.i = i
        self.head = nn.Sequential(
            BasicConv(channel, channel, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(channel, channel, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(channel, channel, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(channel, channel, 3, stride=1, padding=(3 - 1) // 2, relu=False)
        )
        self.plug = nn.Sequential(
            BasicConv(channel, channel, 1, stride=1, padding=(1 - 1) // 2, relu=True),
            BasicConv(channel, channel, 1, stride=1, padding=(1 - 1) // 2, relu=False),
            nn.Sigmoid()
        )

        self.SA0 = CA_attention0(channel)  ## Spatial Attention
        self.CA0 = SA_attention0(channel)  ## Channel Attention

        # self.CA = CA_attention(channel)
        # self.SA = SA_attention(channel)
        self.cat = BasicConv(channel*2, channel, 1, stride=1, padding=(1 - 1) // 2, relu=False)
        # self.conv_end = BasicConv(64, 64, 3, stride=1, padding=(3 - 1) // 2, relu=False)

    def forward(self, y):
        g, g0, x = y
        up_g = self.head(g)
        sa_branch = self.SA0(up_g)
        ca_branch = self.CA0(up_g)
        mix = self.cat(torch.cat((sa_branch, ca_branch), 1))
        next_g = torch.mul(mix, self.plug(x)) + g

        # if (self.i+1) % 32 == 0:
        #     # next_g = self.conv_end(next_g) + g0
        #     # next_g = next_g + g0
        #     g0 = next_g
        return next_g, g0, x

class Restorer(nn.Module):
    def __init__(self):
        super(Restorer, self).__init__()
        num_crb = 16
        para = 1

        n_feats = 64
        kernel_size = 3
        reduction = 16
        inp_chans = 1  # 4 RGGB channels, and 4 Variance maps
        act = nn.ReLU(inplace=True)


        # self.draw_conv1 = BasicConv(para, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=False)


        modules_head = [
            BasicConv(n_feats*2, n_feats, 1, stride=1, padding=(1 - 1) // 2, relu=False)
            # BasicConv(n_feats, n_feats, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        ]

        modules_body = [
            attention(n_feats, i) \
            for i in range(num_crb)]

        modules_tail = [
            # BasicConv(n_feats, n_feats, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=True),
            BasicConv(n_feats, n_feats, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        ]

        self.head_x0 = BasicConv(n_feats, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=False)


        self.head_noisy = BasicConv(inp_chans*4, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=False)
        self.head_xi = BasicConv(n_feats, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=False)
        self.encoder = nn.Sequential(*[
            BasicConv(n_feats, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(n_feats, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(n_feats, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(n_feats, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=False),
        ])

        self.m_down = PixelUnShuffle(upscale_factor=2)
        self.m_up = nn.PixelShuffle(2)
        self.input_cat = nn.Sequential(*modules_head)
        self.f0noise_cat = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.Conv = BasicConv(n_feats, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=False)
        self.glb = nn.Sequential(
            BasicConv(n_feats, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(n_feats, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=False),
            nn.Sigmoid()
        )
        self.tail = nn.Sequential(*modules_tail)
        # self.alpha = nn.Parameter(torch.Tensor(1).fill_(1))
        # self.beta = nn.Parameter(torch.Tensor(1).fill_(1))

    def forward(self, feature_maps, f0, noisy_map):
        feature_maps = self.head_xi(feature_maps)
        f0 = self.head_x0(f0)
        noisy_map = self.head_noisy(self.m_down(noisy_map))
        cat_f0noise = self.f0noise_cat(torch.cat((f0, noisy_map), dim=1))
        cat_feature_maps = self.input_cat(torch.cat((feature_maps, cat_f0noise), dim=1))
        noisy_map = self.encoder(noisy_map)
        inputs = [cat_feature_maps, cat_feature_maps, noisy_map]
        f, _, _ = self.body(inputs)
        f = torch.mul(self.Conv(f), self.glb(noisy_map))
        # f = self.Conv(f)
        return self.tail(f+feature_maps)

class line(nn.Module):
    def __init__(self):
        super(line, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor(1).fill_(1))
        self.beta = nn.Parameter(torch.Tensor(1).fill_(1))

    def forward(self, x, y):
        # a=self.alpha
        # b=self.beta
        return x * self.alpha + y * self.beta
        # return x  + y

class DN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, para=1, T=4):
        super(DN, self).__init__()
        self.head = nn.Sequential(
            BasicConv(in_nc*4, nf, 3, stride=1, padding=(3 - 1) // 2, relu=False)
        )

        self.m_down = PixelUnShuffle(upscale_factor=2)
        self.m_up = nn.PixelShuffle(2)

        self.T = T
        self.C = Corrector()
        self.P = Predictor(in_nc=3, nf=nf)
        self.F = Restorer()

        self.tail = nn.Sequential(
            BasicConv(nf, out_nc*4, 3, stride=1, padding=(3 - 1) // 2, relu=False)
        )

        self.a = nn.Parameter(torch.ones(1) * 1)
        self.b = nn.Parameter(torch.ones(1) * 1)

    def forward(self, noisyImage):
        M0 = self.head(self.m_down(noisyImage))
        n0, P0 = self.P(noisyImage)
        M1 = self.a*self.F(M0, M0, n0) + self.b*M0
        outs = []
        # outs.append(n0)

        for i in range(self.T):
            n0 = n0 + self.C(n0, P0, M1)
            M1 = self.a*self.F(M1, M0, n0) + self.b*M0
        return self.m_up(self.tail(M1))

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                #torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.xavier_uniform_(m.weight.data)
                #torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
