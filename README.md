
# DSRNet

### Prerequisites
Ubuntu 18.04\
Python==3.8.3\
Torch==1.8.2+cu111\
Torchvision==0.9.2+cu111\

### Dataset
For all datasets, they should be organized in below's fashion:
```
|__dataset_name
   |__images: xxx.jpg ...
   |__GT : xxx.png ...
```
Suppose we use DUTS-TR for training, the overall folder structure should be:
```
|__dataset
    |__DUTS
       |__DUTS-TR
          |__DUTS-TR-Image: xxx.jpg ...
          |__DUTS-TR-Mask : xxx.png ...
   |__benchmark
      |__ECSSD
         |__images: xxx.jpg ...
         |__GT : xxx.png ...
      |__HKU-IS
         |__images: xxx.jpg ...
         |__GT : xxx.png ...
      ...
```
[**ECSSD**](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html) || [**HKU-IS**](https://i.cs.hku.hk/~gbli/deep_saliency.html) || [**DUTS-TE**](http://saliencydetection.net/duts/) || [**DUT-OMRON**](http://saliencydetection.net/dut-omron/) || [**PASCAL-S**](http://cbi.gatech.edu/salobj/)

### Train & Test
**Firstly, make sure you have enough GPU RAM**.\
With default setting (batchsize=8), 24GB RAM is required, but you can always reduce the batchsize to fit your hardware.

Default values in option.py are already set to the same configuration as our paper, so \
after setting the ```--dataset_root``` flag in **option.py**, to train the model (default dataset: DUTS-TR), simply:
```
python main.py --GPU_ID 0
```
to test the model located in the **ckpt** folder (default dataset: DUTS-TE), simply:
```
python main.py --test_only --pretrain "best.pt" --GPU_ID 0
```
If you want to train/test with different settings, please refer to **option.py** for more control options.\
Currently only support training on single GPU.

### Pretrain Model & Pre-calculated Saliency Map
Pre-calculated saliency map: [[Baidu 8a3a]](https://pan.baidu.com/s/11qHLBFnn-bhGH0gj3soQtQ)

Pre-trained model on ECSSD: [[Baidu]](https)

### Evaluation
Firstly, obtain predictions via
```
python main.py --test_only --pretrain "xxx/best.pt" --GPU_ID 0 --save_result --save_msg "result"
```
Output will be saved in `./output/abc` if you specified the **save_msg** flag.

For *PR curve* and *F curve*, we use the code provided by this repo: [[PySODEvalToolkit]](https://github.com/lartpang/PySODEvalToolkit)\
For *MAE*, *F measure* and *S score*, we use the code provided by this repo: [[VST, ICCV-2021]](https://github.com/nnizhang/VST)

### Evaluation Results
#### Quantitative Evaluation
<img src="" alt="drawing" width="1200"/>
<img src="" alt="drawing" width="1200"/>

#### Qualitative Evaluation
<img src="" alt="drawing" width="1200"/>

