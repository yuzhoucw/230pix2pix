import torch
import torch.nn as nn


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



