# Adaptive Aggregation of Arbitrary Online Trackers with a Regret Bound

## Experts

* [DaSiamRPN](https://arxiv.org/abs/1808.06048)[<https://github.com/foolwood/DaSiamRPN>,<https://github.com/songheony/DaSiamRPN>]<sup>[1]</sup>
* [ECO](https://arxiv.org/abs/1611.09224)[<https://github.com/martin-danelljan/ECO>]<sup>[2]</sup>
* [MCCT](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Multi-Cue_Correlation_Filters_CVPR_2018_paper.pdf)[<https://github.com/594422814/MCCT>]<sup>[2]</sup>
* [SiamDW](https://arxiv.org/abs/1901.01660)[<https://github.com/cvpr2019/deeper_wider_siamese_trackers>]
* [SiamFC](https://arxiv.org/abs/1606.09549)[<https://github.com/huanglianghua/siamfc-pytorch>]
* [SiamMask](https://arxiv.org/abs/1812.05050)[<https://github.com/foolwood/SiamMask>]
* [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf)[<https://github.com/huanglianghua/siamrpn-pytorch>]

[1] Since the original code of DaSiamRPN is for Python2, we've had to modify the code a little bit to be compatible with Python3.  
[2] To run Matlab scripts in Python, we've used [transplant](https://github.com/bastibe/transplant) which is much faster than official Matlab API

## Datasets

* [OTB2015](https://ieeexplore.ieee.org/document/7001050)[<http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html>]
* [TColor128](https://ieeexplore.ieee.org/document/7277070)[<http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html>]
* [VOT2018](https://link.springer.com/chapter/10.1007/978-3-030-11009-3_1)[<http://www.votchallenge.net/>]<sup>[3]</sup>

[3] VOT2018 is evaluated in unsupervised experiment as same as other datasets.

## Frameworks

* got10k-toolkit[<https://github.com/got-10k/toolkit>] for tracking datasets.
* pysot-toolkit[<https://github.com/StrangerZhang/pysot-toolkit>] for evaluating trackers.

## Requirements

```sh
conda create -n [ENV_NAME] python=[PYTHON_VERSION>=3.6]
conda install pytorch torchvision cudatoolkit=[CUDA_VERSION] -c pytorch
conda install pyzmq
```

## How to run

```sh
git clone https://github.com/songheony/AAA-WACV
mkdir AAA-WACV/external
cd AAA-WACV/external
git clone [FRAMEWORK_GIT]
git clone [EXPERT_GIT]
conda activate [ENV_NAME]
python tracker.py -e [TRACKER_NAME]
python aggregate.py -t [TRACKERS_NAME] -d [DATASETS_NAME]
```

1. Clone this repository and make external directory.

2. Clone experts who you want to hire.<sup>[4]</sup>

3. Run tracker as a client with zmq.

4. Run our tracker a server with zmq.

[4] Depending on the expert, you may need to install additional subparty libraries such as opencv.

## Author

👤 **Heon Song**

* Github: [@songheony](https://github.com/songheony)
* Contact: songheony@gmail.com

## Show your support

Give a ⭐️ if this project helped you!

***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)
