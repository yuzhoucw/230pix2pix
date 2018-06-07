from time import time
import numpy as np
import matplotlib.pyplot as plt

# from sklearn import datasets
from sklearn.manifold import TSNE
from PIL import Image
import dataloader
from mpl_toolkits.mplot3d import Axes3D
from torchvision import transforms

def load_image(filename, transform = False):
    img = Image.open(filename)

    if transform == True:
        transform = transforms.Compose([
            transforms.Resize(256, Image.BICUBIC), # resize to crop size directly
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])
        img = transform(img)

    data = np.asarray(img, dtype = 'int32').flatten()

    # print(data.shape)
    return data

def get_tsne(n_img):
    set1 = []
    # load gt
    for i in range(n_img):
        set1.append(load_image('test/testB/%s_B.jpg'% (i+1), transform = True))

    # load paper UNET
    for i in range(n_img):
        set1.append(load_image('test/pix_orig/%s_fake_B.png'% (i+1), transform = True))

    # load our test data
    for dataset_name in ['unet_200','res6_200','res9_200']:
        for i in range(n_img):
            set1.append(load_image('test/%s/test_%s.png'% (dataset_name, i+1)))

    print("Calculating TSNE...")
    tsne = TSNE(n_components=3, init='pca', random_state=0)
    result = tsne.fit_transform(set1)

    x_min, x_max = np.min(result, 0), np.max(result, 0)
    result = (result - x_min) / (x_max - x_min)
    # print(result.shape)

    return result


def main():

    n = 20
    result = get_tsne(n)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    labels = ['testB', 'paper UNET', 'unet_200','res6_200','res9_200']
    markers = ['o','^','*', 's','>', 'D']

    for j in range(5):
        x = result[j*n:(j+1)*n, 0]
        y = result[j*n:(j+1)*n, 1]
        z = result[j*n:(j+1)*n, 2]
        ax.scatter(x, y, z, label=labels[j], marker=markers[j], s = 80)

    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


if __name__ == '__main__':
        main()
