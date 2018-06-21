# Pix2pix in Pytorch
This project is based on [Image-to-Image Translation with Conditional Adversarial Networks] (https://arxiv.org/pdf/1611.07004v1.pdf).

![Generated images](https://github.com/yuzhoucw/230pix2pix/blob/master/imgs/generated.png)


## Prerequisites
- Python 3.6.x
- [PyTorch 0.4.x & torchvision](http://pytorch.org/)


## Dataset
Maps dataset can be downloaded from original project [CycleGAN and pix2pix in PyTorch] (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
Should choose the CycleGAN one in the format of
.
├── datasets
|   ├── maps
|   |   ├── trainA             # Training
|   |   ├── trainB
|   |   ├── valA               # Validation
|   |   ├── valB
|   |   ├── testA              # Test
|   |   ├── testB


## Training

### Train model from scratch

```bash
python main.py --mode train --data_dir [data_directory] --n_epoch 200 --G cyc --D cyc --gan_loss MSE
```
Default is "./datasets/maps/". Source (A) and target (B) images should be in folders trainA/trainB, valA/valB, testA/testB separately.

### Train using pretrained model
```bash
python main.py --mode train --pretrain_path ./checkpoints/xxx/xxx.pt
```
Need to provide same configs/options when continue to train a model.

### Plot stats from train.json
```bash
python plot.py --dir ./checkpoints/xxx
```
It will look for train.json in the directory and output plots as result.png.

![Loss](https://github.com/yuzhoucw/230pix2pix/blob/master/imgs/loss.png)

### See more options available
```bash
python main.py -h
```

## Testing
```bash
python main.py --mode test --pretrain_path ./checkpoints/xxx/xxx.pt
```
This generates all images from test set and save them to ./checkpoints/xxx/images/test/.


![MSE](https://github.com/yuzhoucw/230pix2pix/blob/master/imgs/mse.png)
![tsne](https://github.com/yuzhoucw/230pix2pix/blob/master/imgs/tsne.png)
