import torch
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

parser = argparse.ArgumentParser()
parser.add_argument('--gen_dir', default='./checkpoints/unit_map/', type=str)
parser.add_argument('--gt_dir', default='./datasets/maps/testB/', type=str)
parser.add_argument('--crop', default=256, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--unaligned', default=False, type=bool)
parser.add_argument('--device_id', default=0, type=int)


class GANDataset(Dataset):

    # Initial logic here, including reading the image files and transform the data
    def __init__(self, rootA, rootB, transform=None, unaligned=False, device=None):
        # initialize image path and transformation
        sortedA = sorted(os.listdir(rootA), key=lambda name: int(name.split('.')[0].split('_')[0]))
        sortedB = sorted(os.listdir(rootB), key=lambda name: int(name.split('.')[0].split('_')[0]))
        self.image_pathsA = list(map(lambda x: os.path.join(rootA, x), sortedA))
        self.image_pathsB = list(map(lambda x: os.path.join(rootB, x), sortedB))

        self.transform = transform
        self.unaligned = unaligned
        self.device = device

    # override to support indexing
    def __getitem__(self, index):

        image_pathA = self.image_pathsA[index]
        imageA = Image.open(image_pathA).convert('RGB')

        # unaligned the paired images if needed
        index_B = index
        if self.unaligned:
            index_B = random.randint(0, len(self.image_pathsB) - 1)
            image_pathB = self.image_pathsB[index_B]
        else:
            image_pathB = self.image_pathsB[index]

        imageB = Image.open(image_pathB).convert('RGB')

        # transform the images if needed
        if self.transform is not None:
            imageA = self.transform(imageA)
            imageB = self.transform(imageB)

        # convert to GPU tensor
        if self.device is not None:
            imageA = imageA.to(self.device)
            imageB = imageB.to(self.device)

        return imageA, imageB, index + 1, index_B + 1

    # returns the number of examples we read
    def __len__(self):
        return max(len(self.image_pathsA), len(self.image_pathsB))  # of how many examples we have


## return - DataLoader, batch dimension in (batch_size, channel, H, W)
def get_dataloader(image_pathA, image_pathB, batch_size, crop, unaligned=False, device=None, shuffle=True, num_workers=0):

    transform = transforms.Compose([
        transforms.Resize(crop, Image.BICUBIC),  # resize to crop size directly
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_dataset = GANDataset(image_pathA, image_pathB, transform, unaligned, device)

    return DataLoader(dataset=batch_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


"""peak signal to noise ratio score"""
def PSNR(img_gen, img_gt):
    mse = ((img_gen-img_gt)**2).mean()
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(255.0 / torch.sqrt(mse))

"""MSE score"""
def MSE(img_gen, img_gt):
    mse = ((img_gen-img_gt)**2).mean()

    return mse


if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device("cuda:%d" % args.device_id if torch.cuda.is_available() else "cpu")

    dataloader = get_dataloader(args.gen_dir, args.gt_dir,
                                crop=args.crop, batch_size=1,
                                unaligned=args.unaligned, device=device,
                                num_workers=args.num_workers, shuffle=False)

    PSNR_scores = []
    MSE_scores = []
    for i, (img_A, img_B, index_A, index_B) in enumerate(dataloader):
        # print(len(image))
        PSNR_score = PSNR(img_A, img_B)
        MSE_score = MSE(img_A, img_B)

        PSNR_scores.append(PSNR_score)
        MSE_scores.append(MSE_score)

    print('The average PSNR score of the generated images are: ', torch.stack(PSNR_scores).mean())
    print('The average MSE score of the generated images are: ', torch.stack(MSE_scores).mean())