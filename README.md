# Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement


We relese our testing code first. The Zero-DCE is in the product, so we may release the training codes late. 

You can find more details here: https://li-chongyi.github.io/Proj_Zero-DCE.html. Have fun!

The implementation of Zero-DCE is for non-commercial use only. 

# Pytorch
Pytorch implementation of Zero-DCE

## Requirements
1. Python 3 
2. Pytorch

Zero-DCE does not need special configurations. Just basic environment. 

### Folder structure
Download the Zero-DCE_code first.
The following shows the basic folder structure.
```

├── data
│   ├── test_data # testing data. You can make a new folder for your testing data, like LIME, MEF, and NPE.
│   │   ├── LIME 
│   │   └── MEF
│   │   └── NPE
│   └── train_data # will release soon
├── lowlight_test.py # testing code
├── model.py # Zero-DEC network
├── dataloader.py
├── snapshots
│   ├── Epoch99.pth #  A pre-trained snapshot (Epoch99.pth)
```
### Test
```
python lowlight_test.py 
```
The script will process the images in the sub-folders of "test_data" folder and make a new folder "result" in the "data". You can find the enhanced images in the "result" folder.

## Bibtex

```
@inproceedings{Zero-DCE,
 author = {Guo, Chunle Guo and Li, Chongyi and Guo, Jichang and Loy, Chen Change and Hou, Junhui and Kwong, Sam and Cong, Runmin},
 title = {Zero-reference deep curve estimation for low-light image enhancement},
 booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)},
 pages    = {1780-1789},
 month = {June},
 year = {2020}
}
```

(Full paper: http://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)

## Contact
If you have any questions, please contact Chongyi Li at lichongyi25@gmail.com or Chunle Guo at guochunle@tju.edu.cn.


