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

        self.gan_loss_fn = torch.nn.MSELoss()
        self.cycle_loss_fn = torch.nn.L1Loss()

        self.lambda_A = args.lambda_A
        self.lambda_B = args.lambda_B


    def to(self, device):
        self.G_A.to(device)
        self.G_B.to(device)
        self.D_A.to(device)
        self.D_B.to(device)

    def train(self, input):
        self.G_A.train()
        self.G_B.train()
        self.D_A.train()
        self.D_B.train()

        A, B = input

        B_gen = self.G_A(A)
        A_cyc = self.G_B(B_gen)
        A_gen = self.G_B(B)
        B_cyc = self.G_A(A_gen)

        ############################
        # G loss
        ############################

        # GAN loss D_A(G_A(A))
        loss_G_A = self.gan_loss(self.D_A(B_gen), 1)
        # GAN loss D_B(G_B(B))
        loss_G_B = self.gan_loss(self.D_B(A_gen), 1)
        # Forward cycle loss
        loss_cyc_A = self.cycle_loss_fn(A_cyc, A) * self.lambda_A
        # Backward cycle loss
        loss_cyc_B = self.cycle_loss_fn(B_cyc, B) * self.lambda_B
        # Combine
        loss_G = loss_G_A + loss_G_B + loss_cyc_A + loss_cyc_B

        loss_G.backward(retain_graph=True)
        self.optimizer_G.step()

        ############################
        # D loss
        ############################

        #TODO: redo foward pass using updated weights or use retain graph?
        #TODO: D fake loss use pool
        #
        print(self.D_B(A))

        loss_D_A_real = self.gan_loss(self.D_A(B), 1)
        loss_D_A_fake = self.gan_loss(self.D_A(B_gen), 0)
        loss_D_A = loss_D_A_real + loss_D_A_fake

        loss_D_B_real = self.gan_loss(self.D_B(A), 1)
        loss_D_B_fake = self.gan_loss(self.D_B(A_gen), 0)
        loss_D_B = loss_D_B_real + loss_D_B_fake

        #TODO can add?
        loss_D = (loss_D_A + loss_D_B) * 0.5
        loss_D.backward()
        self.optimizer_D.step()

        # TODO batch avg?

        # # loss
        # B_gen, loss_G_A, loss_G_B, loss_D_A, loss_D_B =  self.compute_loss(A, B)
        #
        # # backward and update
        # self.optimizer_G.zero_grad()
        # loss_G = loss_G_A + loss_G_B
        # loss_G.backward()
        # self.optimizer_G.step()
        #
        # self.optimizer_D.zero_grad()
        # loss_D = loss_D_A + loss_D_B
        # loss_D.backward()
        # # loss_D_B.backward()
        # self.optimizer_D.step()

        return {'G': loss_G, 'G_A': loss_G_A, 'G_B': loss_G_B, 'Cyc_A': loss_cyc_A, 'Cyc_B': loss_cyc_B,
                'D': loss_D, 'D_A': loss_D_A, 'D_B': loss_D_B}

    def eval(self, input):
        self.G_A.eval()
        self.G_B.eval()
        self.D_A.eval()
        self.D_B.eval()

        A, B = input

        B_gen = self.G_A(A)
        A_cyc = self.G_B(B_gen)
        A_gen = self.G_B(B)
        B_cyc = self.G_A(A_gen)

        ############################
        # G loss
        ############################

        # GAN loss D_A(G_A(A))
        loss_G_A = self.gan_loss(self.D_A(B_gen), 1)
        # GAN loss D_B(G_B(B))
        loss_G_B = self.gan_loss(self.D_B(A_gen), 1)
        # Forward cycle loss
        loss_cyc_A = self.cycle_loss_fn(A_cyc, A) * self.lambda_A
        # Backward cycle loss
        loss_cyc_B = self.cycle_loss_fn(B_cyc, B) * self.lambda_B
        # Combine
        loss_G = loss_G_A + loss_G_B + loss_cyc_A + loss_cyc_B


        ############################
        # D loss
        ############################

        loss_D_A_real = self.gan_loss(self.D_A(B), 1)
        loss_D_A_fake = self.gan_loss(self.D_A(B_gen), 0)
        loss_D_A = loss_D_A_real + loss_D_A_fake

        print(self.D_B(A))
        loss_D_B_real = self.gan_loss(self.D_B(A), 1)
        loss_D_B_fake = self.gan_loss(self.D_B(A_gen), 0)
        loss_D_B = loss_D_B_real + loss_D_B_fake

        loss_D = (loss_D_A + loss_D_B) * 0.5

        # TODO save B_gen

        return {'G': loss_G, 'G_A': loss_G_A, 'G_B': loss_G_B, 'Cyc_A': loss_cyc_A, 'Cyc_B': loss_cyc_B,
                'D': loss_D, 'D_A': loss_D_A, 'D_B': loss_D_B}

    def compute_loss(self, A, B):
        B_gen = self.G_A(A)
        A_cyc = self.G_B(B_gen)
        A_gen = self.G_B(B)
        B_cyc = self.G_A(A_gen)


        loss_G_A = self.gan_loss(self.D_A(B_gen), 1) + self.cycle_loss_fn(A_cyc, A) * self.lambda_A


        loss_G_B = self.gan_loss(self.D_B(A_gen), 1) + self.cycle_loss_fn(B_cyc, B) * self.lambda_B

        loss_D_A = self.gan_loss(self.D_A(B), 1) + self.gan_loss(self.D_A(B_gen), 0) * 0.5 # TODO
        loss_D_B = self.gan_loss(self.D_B(A), 1) + self.gan_loss(self.D_B(A_gen), 0)
        return B_gen, loss_G_A, loss_G_B, loss_D_A, loss_D_B

    def gan_loss(self, out, label):
        return self.gan_loss_fn(out, torch.ones_like(out) if label else torch.zeros_like(out))

    def load_state(self, state, lr=None):
        print('Using pretrained model...')
        self.G_A.load_state_dict(state['G_A'])
        self.G_B.load_state_dict(state['G_B'])
        self.D_A.load_state_dict(state['D_A'])
        self.D_B.load_state_dict(state['D_B'])
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
        return {'G_A': self.G_A.state_dict(),
                'G_B': self.G_B.state_dict(),
                'D_A': self.D_A.state_dict(),
                'D_B': self.D_B.state_dict(),
                'optimG': self.optimizer_G.state_dict(),
                'optimD': self.optimizer_D.state_dict()}

