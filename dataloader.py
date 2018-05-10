import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image



class GANDataset(Dataset):

    # Initial logic here, including reading the image files and transform the data
    def __init__(self, rootA, rootB, transform=None):
        # initialize image path and transformation
        self.image_pathsA = list(map(lambda x: os.path.join(rootA, x), os.listdir(rootA)))
        self.image_pathsB = list(map(lambda x: os.path.join(rootB, x), os.listdir(rootB)))
        self.transform = transform

    # override to support indexing
    def __getitem__(self, index):
        
        image_pathA = self.image_pathsA[index]
        imageA = Image.open(image_pathA).convert('RGB')
        
        image_pathB = self.image_pathsB[index]    
        imageB = Image.open(image_pathB).convert('RGB')
        
        # transform the images if needed
        if self.transform is not None:
            imageA = self.transform(imageA)
            imageB = self.transform(imageB)

        return imageA, imageB

    # returns the number of examples we read
    def __len__(self):
        return max(len(self.image_pathsA), len(self.image_pathsB))  # of how many examples we have


## return - DataLoader, batch dimension in (batch_size, channel, H, W)
def get_dataloader(image_pathA, image_pathB, batch_size, image_size=(256, 256)):
    transform = transforms.Compose([
        # resize PIL image to given size
        transforms.Resize(image_size),
        # convert image input into torch tensor
        transforms.ToTensor(),
        # normalize image with mean and standard deviation
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    batch_dataset = GANDataset(image_pathA, image_pathB, transform)

    return DataLoader(dataset=batch_dataset, batch_size=batch_size, shuffle=True)

