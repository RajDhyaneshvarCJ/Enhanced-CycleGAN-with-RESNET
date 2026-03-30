# Enhanced-CycleGAN-with-RESNET

An enhanced CycleGAN implementation for the Kaggle ["I'm Something of a Painter Myself"](https://www.kaggle.com/competitions/gan-getting-started) competition. Translates photographs into Monet-style paintings using unpaired image-to-image translation.

**Public leaderboard score: 61.33 MiFID**

![Training Loss Animation](assets/training_loss_animation.gif)

## Sample Results

| | | | |
|:---:|:---:|:---:|:---:|
| ![](samples/image_06849.jpg) | ![](samples/image_06884.jpg) | ![](samples/image_06922.jpg) | ![](samples/image_06939.jpg) |
| ![](samples/image_06949.jpg) | ![](samples/image_06834.jpg) | ![](samples/image_06876.jpg) | ![](samples/image_06904.jpg) |
| ![](samples/image_06965.jpg) | ![](samples/image_06871.jpg) | ![](samples/image_06989.jpg) | ![](samples/image_06857.jpg) |

## Architecture

**Generator:** ResNet with 9 residual blocks (encoder-decoder with skip connections)
- 7x7 conv (64ch, stride 1) → 3x3 conv (128ch, ↓2) → 3x3 conv (256ch, ↓2) → 9 ResNet blocks → 2 deconv layers (↑2) → 7x7 conv (3ch) → tanh
- Instance Normalization throughout
- Learns "delta" updates rather than full reconstruction

**Discriminator:** PatchGAN
- Conv(64, ↓2) → Conv(128, ↓2) → Conv(256, ↓2) → Conv(512, stride 1) → Conv(1) patch map
- 4x4 kernels, LeakyReLU, Instance Normalization from second layer onward

## Loss Functions

```
L_G = L_adv + L_cyc + L_id

L_adv = MSE(D(G(x)), 1)                              LSGAN adversarial
L_cyc = 10.0 × (|x - F(G(x))| + |y - G(F(y))|)      Cycle consistency
L_id  = 10.0 × 0.5 × (|y - G(y)| + |x - F(x)|)      Identity
```


## Training Details

| Parameter | Value |
|---|---|
| Image size | 256×256 |
| Batch size | 4 |
| Epochs | 25 |
| Steps per epoch | 500 |
| Optimizer | Adam (lr=2e-4, β₁=0.5) |
| LR schedule | Constant |
| Augmentation | Resize 286 → crop 256, horizontal flip |
| Framework | TensorFlow 2.18 |
| Output | 7,000 generated images |

## Key Implementation Choices

- **LSGAN** over binary cross-entropy for stable, informative gradients
- **Instance Normalization** over Batch Normalization for per-image style independence
- **Strong identity loss** (λ_id=0.5) to prevent unwanted color tinting
- **Minimal augmentation** (spatial only, no color shifts) to let the generator learn Monet's palette
- **Persistent gradient tape** for consistent updates across all four networks

## Run

The notebook runs as-is on Kaggle with GPU. It reads from the competition dataset:

```
/kaggle/input/gan-getting-started/monet_tfrec/*.tfrec
/kaggle/input/gan-getting-started/photo_tfrec/*.tfrec
```

Generates 7,000 Monet-style images and zips them for submission.

## Article

A detailed writeup covering the architecture, loss functions, training dynamics, and what the training logs actually mean is available on [Medium](https://medium.com/@dhyaneshmrthy/why-gans-are-hard-and-why-most-fail-quietly-80961a92c715).

## References

1. Zhu et al. (2017). [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
2. Mao et al. (2017). [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)
3. Isola et al. (2017). [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
4. Torbunov et al. (2023). [Rethinking CycleGAN: Improving Quality of GANs for Unpaired Image-to-Image Translation](https://arxiv.org/abs/2303.16280)
5. Parmar, G., Park, T., Narasimhan, S., & Zhu, J.Y. (2024). [One-Step Image Translation with Text-to-Image Models](https://arXiv.org/abs/2403.12036)

## License

MIT
