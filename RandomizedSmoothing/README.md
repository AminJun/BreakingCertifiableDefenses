# Breaking Ramdomized Gaussian Smoothing

In this repo, we provide out implementation for attacking the [Certifiable Adversarial Robustness via Randomized Smoothing](https://arxiv.org/abs/1902.02918). 
We would like to note that most of this implementation is based on their [official implementation for Randomized Gaussian Smoothing](https://github.com/locuslab/smoothing), so to them, we give the credit. 

### Installation
Firstly, install the required libraries from `requirements.txt`. You may simply use:
```
pip install -r requirements.txt
```
Note that in the `requirements.txt`, two following lines have been commented:
```
#torch==1.3.1+cu92
#torchvision==0.4.2+cu92
```
Uncomment them before installing the requirements if you are using CUDA9.
Otherwise, keep them commented and see [official pytorch installation](https://pytorch.org/get-started/locally/) for more details.

### Run
To run the experiments, first of all you need the pretrained models from the [official implementation for Randomized Gaussian Smoothing](https://drive.google.com/file/d/1h_TpbXm5haY5f-l4--IKylmdz6tvPoR4/view?usp=sharing) and put them into the models directory. 

Then for `CIFAR-10` experiments, simply run:
```
source cifar10.sh &
```

For `ImageNet` experiments, first of all you need to point the `imagenet.sh` to the directory for `ImageNet` dataset:
```
export IMAGENET_DIR="PUT Directory Address for ImageNet dataset here!"
```
Then simply run:
```
source imagenet.sh & 
```
