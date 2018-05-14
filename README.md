#Pix2pix in Pytorch

## Train model from scratch

```bash
python main.py --mode train --n_epoch 200
```

## Use pretrained model
```bash
python main.py --mode [train|test] --pretrain_path ./checkpoints/xxx/xxx.pt
```

## Plot stats from train.json
```bash
python plot.py --dir ./checkpoints/xxx
```
It will look for train.json in the directory and output plots as result.png.