# Adaptive Aggregation of Arbitrary Online Trackers <br/> with a Regret Bound

[Adaptive Aggregation of Arbitrary Online Trackers with a Regret Bound](http://openaccess.thecvf.com/content_WACV_2020/html/Song_Adaptive_Aggregation_of_Arbitrary_Online_Trackers_with_a_Regret_Bound_WACV_2020_paper.html) [[pdf](http://openaccess.thecvf.com/content_WACV_2020/papers/Song_Adaptive_Aggregation_of_Arbitrary_Online_Trackers_with_a_Regret_Bound_WACV_2020_paper.pdf)] [[poster](https://drive.google.com/open?id=1L7rOHWUWEVk92Vfz8SwroKVDAIk9PjMs)]

Heon Song, Daiki Suehiro, and Seiichi Uchida

The IEEE Winter Conference on Applications of Computer Vision(WACV), 2020

> We propose an online visual-object tracking method that is robust even in an adversarial environment, where various disturbances may occur on the target appearance, etc. The proposed method is based on a delayed-Hedge algorithm for aggregating multiple arbitrary online trackers with adaptive weights. The robustness in the tracking performance is guaranteed theoretically in term of "regret" by the property of the delayed-Hedge algorithm. Roughly speaking, the proposed method can achieve a similar tracking performance as the best one among all the trackers to be aggregated in an adversarial environment. The experimental study on various tracking tasks shows that the proposed method could achieve state-of-the-art performance by aggregating various online trackers. 

## Tracking examples
| BlurBody | Hand | car1 | motocross1 | leaves |
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
<a href="https://drive.google.com/uc?export=view&id=1LJ4VQ2CPvTeIwSj3sSqolWxYedEdv2VF"><img border="0" alt="BlurBody" src="https://drive.google.com/uc?export=view&id=1NC2oDGns_zYvJ0jL3yqVolMGedj0LCP8" width="200" height="200"></a> | <a href="https://drive.google.com/uc?export=view&id=1ZJHY52iJwABgZ-GJxEBQmqGcD-URQisH"><img border="0" alt="Hand" src="https://drive.google.com/uc?export=view&id=1j3uoCJd8H95nIMZHVzwA_7U4YcTM-EXY" width="200" height="200"></a>  | <a href="https://drive.google.com/uc?export=view&id=1Z8STxs-WzXG9RwNBui74dM4CERzc2_xN"><img border="0" alt="car1" src="https://drive.google.com/uc?export=view&id=1PAS4AwjqcOnsdTHP-mSdQVbK-FkgURxK" width="200" height="200"></a>  | <a href="https://drive.google.com/uc?export=view&id=1n-4pKK4c-0fn_JX-kUEUKz5J9vs2rsid"><img border="0" alt="motocross1" src="https://drive.google.com/uc?export=view&id=1TGemjJJ-SiUXtdzqLqmwVakNWBU-RggM" width="200" height="200"></a>  | <a href="https://drive.google.com/uc?export=view&id=1AvlAUv4JdllBgr7BYlAUTSDEQ6B6bnIm"><img border="0" alt="leaves" src="https://drive.google.com/uc?export=view&id=11DFpa4vOjzzFA73qvJ1lYHCZVatcnWAb" width="200" height="200"></a>

## Experts

* [DaSiamRPN](https://arxiv.org/abs/1808.06048)[<https://github.com/foolwood/DaSiamRPN>,<https://github.com/songheony/DaSiamRPN>]<sup>[1]</sup>
* [ECO](https://arxiv.org/abs/1611.09224)[<https://github.com/martin-danelljan/ECO>]<sup>[2]</sup>
* [MCCT](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Multi-Cue_Correlation_Filters_CVPR_2018_paper.pdf)[<https://github.com/594422814/MCCT>]<sup>[2]</sup>
* [SiamDW](https://arxiv.org/abs/1901.01660)[<https://github.com/cvpr2019/deeper_wider_siamese_trackers>]
* [SiamFC](https://arxiv.org/abs/1606.09549)[<https://github.com/huanglianghua/siamfc-pytorch>]
* [SiamMask](https://arxiv.org/abs/1812.05050)[<https://github.com/foolwood/SiamMask>]
* [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf)[<https://github.com/huanglianghua/siamrpn-pytorch>]

[1] Since the original code of DaSiamRPN is for Python2, we've had to modify the code a little bit to be compatible with Python3.  
[2] To run Matlab scripts in Python, we've used [transplant](https://github.com/bastibe/transplant) which is much faster than official Matlab API.

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
docker run -it --name [TRACKER_NAME] --network [NETWORK_NAME] python tracker.py -e [TRACKER_NAME]
docker run -it --name server --network [NETWORK_NAME] python aggregate.py -t [TRACKERS_NAME] -d [DATASETS_NAME]
```

1. Clone this repository and make external directory.

2. Clone experts who you want to hire.<sup>[4]</sup>

3. [Create network over docker](https://docs.docker.com/network/network-tutorial-overlay/).

4. Run tracker as a server with docker.

5. Run our tracker a client with docker.

[4] Depending on the expert, you may need to install additional subparty libraries such as opencv.

## Author

👤 **Heon Song**

* Github: [@songheony](https://github.com/songheony)
* Contact: songheony@gmail.com

## Citation

```bib
@InProceedings{Song_2020_WACV,
    author = {Song, Heon and Suehiro, Daiki and Uchida, Seiichi},
    title = {Adaptive Aggregation of Arbitrary Online Trackers with a Regret Bound},
    booktitle = {The IEEE Winter Conference on Applications of Computer Vision (WACV)},
    month = {March},
    year = {2020}
}
```
