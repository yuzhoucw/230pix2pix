
import torch
import torch.nn as nn
import torch.nn.functional as F

def norm_relu_layer(out_channel, norm, relu):
    if norm == "batch":
        norm_layer = nn.BatchNorm2d(out_channel)
    elif norm == "instance":
        norm_layer = nn.InstanceNorm2d(out_channel)
    elif norm is None:
        norm_layer = nn.Dropout2d(0)  # Identity
    else:
        raise Exception("Norm not specified!")

    if relu is None:
        relu_layer = nn.ReLU()
    else:
        relu_layer = nn.LeakyReLU(relu, inplace=True)

    return norm_layer, relu_layer


def Conv_Norm_ReLU(in_channel, out_channel, kernel, padding=0, dilation=1, groups=1, stride=1, bias=True,
                   norm="batch", relu=None):
    """
    Convolutional -- Norm -- ReLU Unit
    :param norm: 'batchnorm' --> use BatchNorm2D, 'instancenorm' --> use InstanceNorm2D, 'none' --> Identity()
    :param relu: None -> Use vanilla ReLU; float --> Use LeakyReLU(relu)

    :input (N x in_channel x H x W)
    :return size same as nn.Conv2D
    """
    norm_layer, relu_layer = norm_relu_layer(out_channel, norm, relu)

    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel, padding=padding, stride=stride,
                  dilation=dilation, groups=groups, bias=bias),
        norm_layer,
        relu_layer
    )


def Deconv_Norm_ReLU(in_channel, out_channel, kernel, padding=0, output_padding=0, stride=1, groups=1,
                     bias=True, dilation=1, norm="batch"):
    """
    Deconvolutional -- Norm -- ReLU Unit
    :param norm: 'batchnorm' --> use BatchNorm2D, 'instancenorm' --> use InstanceNorm2D, 'none' --> Identity()
    :param relu: None -> Use vanilla ReLU; float --> Use LeakyReLU(relu)

    :input (N x in_channel x H x W)
    :return size same as nn.ConvTranspose2D
    """
    norm_layer, relu_layer = norm_relu_layer(out_channel, norm, relu=None)
    return nn.Sequential(
        nn.ConvTranspose2d(in_channel, out_channel, kernel, padding=padding, output_padding=output_padding,
                           stride=stride, groups=groups, bias=bias, dilation=dilation),
        norm_layer,
        relu_layer
    )

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, norm="batch", activation = True): # bias default is True in Conv2d
        super(EncoderBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size =kernel_size, stride=stride, padding=padding, bias=bias)
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.norm = norm
        self.activation = activation
        if norm == "batch":
            self.normalization = nn.BatchNorm2d(out_channels)
        if norm == "instance":
            self.normalization = nn.InstanceNorm2d(out_channels)


    def forward(self, x):
        if self.activation:
            x = self.leakyRelu(x)

        x = self.conv(x)

        if self.norm != None:
            x = self.normalization(x)

        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, norm="batch", activation = True, dropout_prob=0.5):
        super(DecoderBlock, self).__init__()

        self.convT = nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size =kernel_size, stride=stride, padding=padding, bias=bias)
        self.relu = nn.ReLU()
        self.dropout_prob = dropout_prob
        self.drop = nn.Dropout2d(dropout_prob)
        self.norm = norm
        self.activation = activation
        if norm == "batch":
            self.normalization = nn.BatchNorm2d(out_channels)
        if norm == "instance":
            self.normalization = nn.InstanceNorm2d(out_channels)


    def forward(self, x):
        if self.activation:
            x = self.relu(x)

        x = self.convT(x)

        if self.norm != None:
            x = self.normalization(x)

        if self.dropout_prob != 0:
            x= self.drop(x)

        return x

class ResidualLayer(nn.Module):
    """
    Residual block used in Johnson's network model:
    Our residual blocks each contain two 3×3 convolutional layers with the same number of filters on both
    layer. We use the residual block design of Gross and Wilber [2] (shown in Figure 1), which differs from
    that of He et al [3] in that the ReLU nonlinearity following the addition is removed; this modified design
    was found in [2] to perform slightly better for image classification.
    """

    def __init__(self, channels, kernel_size, final_relu=False, bias=False, norm='batch'):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.padding = (self.kernel_size[0] - 1) // 2
        self.final_relu = final_relu

        norm_layer, relu_layer = norm_relu_layer(self.channels, norm, relu=None)
        self.layers = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=self.padding, bias=bias),
            norm_layer,
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=self.padding, bias=bias),
            norm_layer
        )

    def forward(self, input):
        # input (N x channels x H x W)
        # output (N x channels x H x W)
        out = self.layers(input)
        if self.final_relu:
            return nn.ReLU(out + input)
        else:
            return out + input

class GeneratorJohnson(nn.Module):
    """
    The Generator architecture in < Perceptual Losses for Real-Time Style Transfer and Super-Resolution >
    by Justin Johnson, et al.
    """

    def __init__(self, in_channels=3, out_channels=3, norm='instance', bias=True):
        super(GeneratorJohnson, self).__init__()
        model = []
        model += [Conv_Norm_ReLU(in_channels, 32, (7, 7), padding=3, stride=1, bias=bias, norm=norm),
                  # c7s1-32
                  Conv_Norm_ReLU(32, 64, (3, 3), padding=1, stride=2, bias=bias, norm=norm),  # d64
                  Conv_Norm_ReLU(64, 128, (3, 3), padding=1, stride=2, bias=bias, norm=norm)]  # d128
        for i in range(6):
            model += [ResidualLayer(128, (3, 3), final_relu=False, bias=bias)]  # R128
        model += [
            Deconv_Norm_ReLU(128, 64, (3, 3), padding=1, output_padding=1, stride=2, bias=bias, norm=norm),
            # u64
            Deconv_Norm_ReLU(64, 32, (3, 3), padding=1, output_padding=1, stride=2, bias=bias, norm=norm),
            # u32
            nn.Conv2d(32, out_channels, (7, 7), padding=3, stride=1, bias=bias),  # c7s1-3
            nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """
        :param input: (N x channels x H x W)
        :return: output: (N x channels x H x W) with numbers of range [-1, 1] (since we use tanh())
        """
        return self.model(input)


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dropout_prob=0.5, norm = "batch", bias = False):
        super(Generator, self).__init__()

        # 8-step encoder
        self.encoder1 = EncoderBlock(in_channels, 64, bias = bias, norm = None, activation=False)
        self.encoder2 = EncoderBlock(64, 128, bias = bias, norm = norm)
        self.encoder3 = EncoderBlock(128, 256, bias = bias, norm = norm)
        self.encoder4 = EncoderBlock(256, 512, bias = bias, norm = norm)
        self.encoder5 = EncoderBlock(512, 512, bias = bias, norm = norm)
        self.encoder6 = EncoderBlock(512, 512, bias = bias, norm = norm)
        self.encoder7 = EncoderBlock(512, 512, bias = bias, norm = norm)
        self.encoder8 = EncoderBlock(512, 512, bias = bias, norm = None)

        # 8-step UNet decoder
        self.decoder1 = DecoderBlock(512, 512, bias = bias, norm = norm)
        self.decoder2 = DecoderBlock(1024, 512, bias = bias, norm = norm, dropout_prob = dropout_prob)
        self.decoder3 = DecoderBlock(1024, 512, bias = bias, norm = norm, dropout_prob = dropout_prob)
        self.decoder4 = DecoderBlock(1024, 512, bias = bias, norm = norm, dropout_prob = dropout_prob)
        self.decoder5 = DecoderBlock(1024, 256, bias = bias, norm = norm)
        self.decoder6 = DecoderBlock(512, 128, bias = bias, norm = norm)
        self.decoder7 = DecoderBlock(256, 64, bias = bias, norm = norm)
        self.decoder8 = DecoderBlock(128, out_channels, bias = bias, norm = None)


    def forward(self, x):
        # 8-step encoder
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(encode1)
        encode3 = self.encoder3(encode2)
        encode4 = self.encoder4(encode3)
        encode5 = self.encoder5(encode4)
        encode6 = self.encoder6(encode5)
        encode7 = self.encoder7(encode6)
        encode8 = self.encoder8(encode7)

        # 8-step UNet decoder
        decode1 = torch.cat([self.decoder1(encode8), encode7],1)
        decode2 = torch.cat([self.decoder2(decode1), encode6],1)
        decode3 = torch.cat([self.decoder3(decode2), encode5],1)
        decode4 = torch.cat([self.decoder4(decode3), encode4],1)
        decode5 = torch.cat([self.decoder5(decode4), encode3],1)
        decode6 = torch.cat([self.decoder6(decode5), encode2],1)
        decode7 = torch.cat([self.decoder7(decode6), encode1],1)
        decode8 = self.decoder8(decode7)
        final = nn.Tanh()(decode8)
        return final


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, norm = "batch", bias = False):
        super(Discriminator, self).__init__()

        # self.out_channels = out_channels

        # 70x70 discriminator
        self.disc1 = EncoderBlock(in_channels * 2, 64, bias = bias, norm = None, activation=False)
        self.disc2 = EncoderBlock(64, 128, bias = bias, norm = norm)
        self.disc3 = EncoderBlock(128, 256, bias = bias, norm = norm)
        self.disc4 = EncoderBlock(256, 512, bias = bias, stride=1, norm = norm)
        self.disc5 = EncoderBlock(512, out_channels, bias = bias, stride=1, norm = norm)

    def forward(self, x, ref):
        d1 = self.disc1(torch.cat([x, ref],1))
        d2 = self.disc2(d1)
        d3 = self.disc3(d2)
        d4 = self.disc4(d3)
        d5 = self.disc5(d4)
        final = nn.Sigmoid()(d5)
        return final



