# Breaking CROWN-IBP Defense
In this repository, we include our code for breaking the [CROWN-IBP method](https://arxiv.org/abs/1906.06316).
Most of the code has been taken from the [official repository](https://github.com/huanzhang12/CROWN-IBP) for the CROWN-IBP paper. 
To them, we give all the credit for implementing the CROWN-IBP method and other methods included in the repository except for the shadow attack part.

# Start
Firstly, install the required libraries from `requirements.txt`. You may simply use:
```
pip install -r requirements.txt 
``` 

To reproduce the results in the [paper](https://openreview.net/forum?id=HJxdTxHYvB), simply run the `run.sh` runner file. 
The runner at first, downloads the pretrained models from the [official repository for pretrained models from Huan Zhangs website](https://download.huan-zhang.com/models/crown-ibp/models_crown-ibp.tar.gz). Simply, run:
```
source run.sh &
```
