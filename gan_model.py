import torch
from model_modules import *

class GANModel:

    def __init__(self, args):
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.G = Generator()
        self.D = Discriminator()

        self.optimizer_G = torch.optim.Adam(self.G.parameters(),
                                            lr=args.lr, betas=(args.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(),
                                            lr=args.lr, betas=(args.beta1, 0.999))

        self.gan_loss_fn = torch.nn.BCELoss()
        self.L1_loss_fn = torch.nn.L1Loss()

        self.lambd = args.lambd



    def to(self, device):
        self.G.to(device)
        self.D.to(device)

    def train(self, input):
        self.G.train()
        self.D.train()

        x, y = input

        ############################
        # D loss
        ############################
        self.optimizer_D.zero_grad()

        gen = self.G(x)
        # real y and x -> 1
        loss_D_real = self.gan_loss(self.D(y, x), 1) / 2
        # gen and x -> 0
        loss_D_fake = self.gan_loss(self.D(gen, x), 0) / 2
        # Combine
        loss_D = (loss_D_real + loss_D_fake)

        loss_D.backward()
        self.optimizer_D.step()

        ############################
        # G loss
        ############################
        self.optimizer_G.zero_grad()

        gen = self.G(x)
        # GAN loss of G
        loss_G_gan = self.gan_loss(self.D(gen, x), 1)
        # L1 loss of G
        loss_G_L1 = self.L1_loss_fn(gen, y) * self.lambd
        # Combine
        loss_G = loss_G_gan + loss_G_L1

        loss_G.backward()
        self.optimizer_G.step()


        return {'G': loss_G, 'G_gan': loss_G_gan, 'G_L1': loss_G_L1,
                'D': loss_D, 'D_real': loss_D_real, 'D_fake': loss_D_fake}

    def eval(self, input):
        self.G.eval()
        self.D.eval()

        x, y = input
        gen = self.G(x)

        ############################
        # D loss
        ############################
        # real y and x -> 1
        loss_D_real = self.gan_loss(self.D(y, x), 1) /2
        # gen and x -> 0
        loss_D_fake = self.gan_loss(self.D(gen, x), 0) /2
        # Combine
        loss_D = (loss_D_real + loss_D_fake)

        ############################
        # G loss
        ############################
        # GAN loss of G
        loss_G_gan = self.gan_loss(self.D(gen, x), 1)
        # L1 loss of G
        loss_G_L1 = self.L1_loss_fn(gen, y)
        # Combine
        loss_G = loss_G_gan + loss_G_L1


        return {'G': loss_G, 'G_gan': loss_G_gan, 'G_L1': loss_G_L1,
                'D': loss_D, 'D_real': loss_D_real, 'D_fake': loss_D_fake}


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

