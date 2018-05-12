import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=False, do_batch_norm=True): # bias default is True in Conv2d
        super(EncoderBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.do_batch_norm = do_batch_norm

    def forward(self, x):
       x = self.conv(x)
       if self.do_batch_norm:
           x = self.bn(x)
       x = self.leakyRelu(x)
       return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, do_batch_norm=True, dropout_prob=0.0):
        super(DecoderBlock, self).__init__()

        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout_prob = dropout_prob
        self.drop = nn.Dropout2d(dropout_prob)
        self.do_batch_norm = do_batch_norm

    def forward(self, x):
       x = self.convT(x)
       if self.dropout_prob != 0:
           x= self.drop(x)
       if self.do_batch_norm:
           x = self.bn(x)
       x = self.relu(x)

       return x

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()

        # 8-step encoder
        self.encoder1 = EncoderBlock(in_channels, 64, do_batch_norm=False)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.encoder5 = EncoderBlock(512, 512)
        self.encoder6 = EncoderBlock(512, 512)
        self.encoder7 = EncoderBlock(512, 512)
        self.encoder8 = EncoderBlock(512, 512, do_batch_norm=False)

        # 8-step UNet decoder
        self.decoder1 = DecoderBlock(512, 512, dropout_prob=0.5)
        self.decoder2 = DecoderBlock(1024, 512, dropout_prob=0.5)
        self.decoder3 = DecoderBlock(1024, 512, dropout_prob=0.5)
        self.decoder4 = DecoderBlock(1024, 512)
        self.decoder5 = DecoderBlock(1024, 256)
        self.decoder6 = DecoderBlock(512, 128)
        self.decoder7 = DecoderBlock(256, 64)
        self.decoder8 = DecoderBlock(128, out_channels)


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
    def __init__(self, in_channels=3, out_channels=1):
        super(Discriminator, self).__init__()

        self.out_channels = out_channels

        # 70x70 discriminator
        self.disc1 = EncoderBlock(in_channels * 2, 64, do_batch_norm=False)
        self.disc2 = EncoderBlock(64, 128)
        self.disc3 = EncoderBlock(128, 256)
        self.disc4 = EncoderBlock(256, 512, stride=1)
        self.conv = nn.Conv2d(in_channels = 512, out_channels = self.out_channels, kernel_size=4, stride=1, padding = 1)


    def forward(self, x, ref):
        d1 = self.disc1(torch.cat([x, ref],1))
        d2 = self.disc2(d1)
        d3 = self.disc3(d2)
        d4 = self.disc4(d3)
        final = self.conv(d4)
        final = nn.Sigmoid()(final)
        return final



