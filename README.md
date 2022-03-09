# Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement

You can find more details here: https://li-chongyi.github.io/Proj_Zero-DCE.html. Have fun!

**The implementation of Zero-DCE is for non-commercial use only.**

We also provide a MindSpore version of our code: https://pan.baidu.com/s/1uyLBEBdbb1X4QVe2waog_g (passwords: of5l).

# Pytorch 
Pytorch implementation of Zero-DCE

## Requirements
1. Python 3.7 
2. Pytorch 1.0.0
3. opencv
4. torchvision 0.2.1
5. cuda 10.0

Zero-DCE does not need special configurations. Just basic environment. 

Or you can create a conda environment to run our code like this:
conda create --name zerodce_env opencv pytorch==1.0.0 torchvision==0.2.1 cuda100 python=3.7 -c pytorch

### Folder structure
Download the Zero-DCE_code first.
The following shows the basic folder structure.
```

├── data
│   ├── test_data # testing data. You can make a new folder for your testing data, like LIME, MEF, and NPE.
│   │   ├── LIME 
│   │   └── MEF
│   │   └── NPE
│   └── train_data 
├── lowlight_test.py # testing code
├── lowlight_train.py # training code
├── model.py # Zero-DEC network
├── dataloader.py
├── snapshots
│   ├── Epoch99.pth #  A pre-trained snapshot (Epoch99.pth)
```
### Test: 

cd Zero-DCE_code
```
python lowlight_test.py 
```
The script will process the images in the sub-folders of "test_data" folder and make a new folder "result" in the "data". You can find the enhanced images in the "result" folder.

### Train: 
1) cd Zero-DCE_code

2) download the training data <a href="https://drive.google.com/file/d/1GAB3uGsmAyLgtDBDONbil08vVu5wJcG3/view?usp=sharing">google drive</a> or <a href="https://pan.baidu.com/s/11-u_FZkJ8OgbqcG6763XyA">baidu cloud [password: 1234]</a>

3) unzip and put the  downloaded "train_data" folder to "data" folder
```
python lowlight_train.py 
```
##  License
The code is made available for academic research purpose only. Under Attribution-NonCommercial 4.0 International License.


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

## TensorFlow Version 
Thanks tuvovan (vovantu.hust@gmail.com) who re-produces our code by TF. The results of TF version look similar with our Pytorch version. But I do not have enough time to check the details.
https://github.com/tuvovan/Zero_DCE_TF
