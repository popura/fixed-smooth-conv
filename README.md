# Fixed Smooth Convolutional Layer
This is an implementation for "Fixed Smooth Convolutional Layer for Avoiding Checkerboard Artifacts in CNNs."

When you use this implementation for your research work,
please cite the following paper.
```
@inproceedings{kinoshita2020fixed,
author = {Kinoshita, Yuma and Kiya, Hitoshi},
booktitle = {Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing},
doi = {10.1109/ICASSP40776.2020.9054096},
isbn = {978-1-5090-6631-5},
month = {May},
pages = {3712--3716},
publisher = {IEEE},
title = {{Fixed Smooth Convolutional Layer for Avoiding Checkerboard Artifacts in CNNs}},
url = {https://ieeexplore.ieee.org/document/9054096/},
year = {2020}
}
```

# Requirements
- Python = "^3.8"
- Pytorch = "^1.7.0"

# Installation

# Usage
You can use a fixed smooth convolutional layer in the same way as normal convolutional layers (e.g. Conv2d, ConvTranspose2d).
```
    from layer import UpSampling2d, DownSampling2d
    
    # For upsampling with a stride of 2
    up_conv = UpSampling2d(in_channels, out_channels, kernel_size, stride=2,
                           padding=0, dilation=1, groups=1,
                           bias=True, padding_mode='zeros',
                           order=0, hold_mode='hold_first', bias_mode='bias_first')
    
    # For downsampling with a stride of 2
    down = DownSampling2d(in_channels, out_channels, kernel_size, stride=2,
                          padding=0, dilation=1, groups=1,
                          bias=True, padding_mode='zeros',
                          order=0, hold_mode='hold_first', bias_mode='bias_first')
    
```
