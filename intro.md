# Efficient Adversarial Contrastive Learning via Robustness-aware Coreset Selection (ICML23 Submission)

In this repo, we provide the code and the script for reproduce the experiemtns in the main paper, including ACL on CIFAR-10, ACL on ImageNet-1K, and SAT on ImgeNet-1K. 

### Dataset preparation
As for preparing ImageNet-1K of $224\times 224$ resolution, we use the following scripts.
```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
```

As for preparing ImageNet-1K of $32 \times 32$ resolution, we use the following scripts.

```
wget https://image-net.org/data/downsample/Imagenet32_train.zip
wget https://image-net.org/data/downsample/Imagenet32_val.zip
```