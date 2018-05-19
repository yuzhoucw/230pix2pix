import torch
from model_modules import *
import numpy as np
import scipy.misc
import os
import itertools

class GANModel:

    def __init__(self, args):
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.G = Generator(dropout_prob = args.dropout, norm = args.norm, bias=args.bias)
        self.D = Discriminator(norm = args.norm, bias=args.bias)

        self.optimizer_G = torch.optim.Adam(self.G.parameters(),
                                            lr=args.lr, betas=(args.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(),
                                            lr=args.lr, betas=(args.beta1, 0.999))

        self.gan_loss_fn = torch.nn.BCELoss()
        self.L1_loss_fn = torch.nn.L1Loss()

        self.lambd = args.lambd
        self.lambd_d = args.lambd_d



    def to(self, device):
        self.G.to(device)
        self.D.to(device)

        for state in itertools.chain(self.optimizer_G.state.values(), self.optimizer_D.state.values()):
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def train(self, input, save, out_dir_img, epoch):
        self.G.train()
        self.D.train()

        x, y = input

        ############################
        # D loss
        ############################
        self.optimizer_D.zero_grad()

        gen = self.G(x)
        # real y and x -> 1
        loss_D_real = self.gan_loss(self.D(y, x), 1) * self.lambd_d
        # gen and x -> 0
        loss_D_fake = self.gan_loss(self.D(gen.detach(), x), 0) * self.lambd_d
        # Combine
        loss_D = loss_D_real + loss_D_fake

        loss_D.backward()
        self.optimizer_D.step()

        # self.save_image((x, gen, y), 'datasets/maps/samples', '2018')
        ############################
        # G loss
        ############################
        self.optimizer_G.zero_grad()

        # gen = self.G(x)
        # GAN loss of G
        loss_G_gan = self.gan_loss(self.D(gen, x), 1)
        # L1 loss of G
        loss_G_L1 = self.L1_loss_fn(gen, y) * self.lambd
        # Combine
        loss_G = loss_G_gan + loss_G_L1

        loss_G.backward()
        self.optimizer_G.step()

        # save image
        if save:
            self.save_image((x, y, gen), out_dir_img, "train_ep_%d" % epoch)

        return {'G': loss_G, 'G_gan': loss_G_gan, 'G_L1': loss_G_L1,
                'D': loss_D, 'D_real': loss_D_real, 'D_fake': loss_D_fake}

    def eval(self, input, save, out_dir_img, epoch):
        self.G.eval()
        self.D.eval()

        x, y = input
        gen = self.G(x)

        # self.save_image((x, gen, y), 'datasets/maps/samples', '2018')

        ############################
        # D loss
        ############################
        # real y and x -> 1
        loss_D_real = self.gan_loss(self.D(y, x), 1) * self.lambd_d
        # gen and x -> 0
        loss_D_fake = self.gan_loss(self.D(gen, x), 0) * self.lambd_d
        # Combine
        loss_D = loss_D_real + loss_D_fake

        ############################
        # G loss
        ############################
        # GAN loss of G
        loss_G_gan = self.gan_loss(self.D(gen, x), 1)
        # L1 loss of G
        loss_G_L1 = self.L1_loss_fn(gen, y) * self.lambd
        # Combine
        loss_G = loss_G_gan + loss_G_L1

        # save image
        if save:
            self.save_image((x, y, gen), out_dir_img, "val_ep_%d" % epoch)

        return {'G': loss_G, 'G_gan': loss_G_gan, 'G_L1': loss_G_L1,
                'D': loss_D, 'D_real': loss_D_real, 'D_fake': loss_D_fake}

    def test(self, images, i, out_dir_img):
        A, B = images
        self.save_image((A, B, self.G(A)), out_dir_img, "test_%d" % i)


    def gan_loss(self, out, label):
        return self.gan_loss_fn(out, torch.ones_like(out) if label else torch.zeros_like(out))

    def load_state(self, state, lr=None):
        print('Using pretrained model...')
        self.G.load_state_dict(state['G'])
        self.D.load_state_dict(state['D'])
        self.optimizer_G.load_state_dict(state['optimG'])
        self.optimizer_D.load_state_dict(state['optimD'])

        # set model lr to new lr
        if lr is not None:
            for param_group in self.optimizer_G.param_groups:
                before = param_group['lr']
                param_group['lr'] = lr
            for param_group in self.optimizer_D.param_groups:
                before = param_group['lr']
                param_group['lr'] = lr
            print('optim lr: before={} / after={}'.format(before, lr))


    def save_state(self):
        return {'G': self.G.state_dict(),
                'D': self.D.state_dict(),
                'optimG': self.optimizer_G.state_dict(),
                'optimD': self.optimizer_D.state_dict()}

    def save_image(self, input, filepath, fname):
        """ input is a tuple of the images we want to compare """
        A, B, gen = input

        # img_A, img_B, img_gen = self.tensor2image(A), self.tensor2image(B), self.tensor2image(gen)

        # merged = self.merge_images(img_A, img_B, img_gen)
        merged = self.tensor2image(self.merge_images(A, B, gen))
        path = os.path.join(filepath, '%s.png' % fname)
        scipy.misc.imsave(path, merged)
        print('saved %s' % path)

    def tensor2image(self, input):
        image_data = input.data
        image = 127.5 * (image_data.cpu().float().numpy() + 1.0)
        return image.astype(np.uint8)

    def merge_images(self, sources, targets, generated):
        # row, _, h, w = sources.shape
        row, _, h, w = sources.size()
        # row = int(np.sqrt(batch_size))
        # merged = np.zeros([3, row * h, w * 3])
        merged = torch.zeros([3, row * h, w * 3])
        for idx, (s, t, g) in enumerate(zip(sources, targets, generated)):
            i = idx
            # i = (idx + 1) // row
            # j = idx % row
            # merged[:, i * h:(i + 1) * h, (j * 2) * w:(j * 2 + 1) * w] = s
            # merged[:, i * h:(i + 1) * h, (j*2+1) * w:(j * 2 + 2) * w] = t
            # merged[:, i * h:(i + 1) * h, (j*2+2) * w:(j * 2 + 3) * w] = c
            merged[:, i * h:(i + 1) * h, 0:w] = s
            merged[:, i * h:(i + 1) * h, w:2 * w] = g
            merged[:, i * h:(i + 1) * h, 2 * w:3 * w] = t
        return merged.permute(1, 2, 0)
