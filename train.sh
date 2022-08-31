#! /bin/bash
{
python train_liif.py --config configs/edsr-baseline.yaml --name edsr-baseline;
python train_liif.py --config configs/lsr-unet.yaml --name lsr-unet;
}
