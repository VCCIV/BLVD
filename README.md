# BLVD

BLVD is a large scale 5D semantics dataset collected by Visual Cognitive Computing and Intelligent Vehicles Lab.

**BLVD: Building A Large-scale 5D Semantics Benchmark for Autonomous Driving**

Jianru Xue, Jianwu Fang, Tao Li, Bohua Zhang, Pu Zhang, Zhen Ye and Jian Dou

![](https://github.com/VCCIV/BLVD/blob/master/1.png)
The task flow of BLVD with 5D semantic annotations: (a) 3D detection, (b) 4D tracking, (c) 5D interactive event recognition and (d) 5D intention prediction.

### About

In autonomous driving community, numerous benchmarks have been established to assist the tasks of 3D/2D object detection, stereo vision, semantic/instance segmentation. However, the more meaningful dynamic evolution of the surrounding objects of ego-vehicle is rarely exploited, and lacks a large-scale dataset platform. To address this, we introduce BLVD, a large-scale 5D semantics benchmark which does not concentrate on the static detection or semantic/instance segmentation tasks tackled adequately before. Instead, BLVD aims to provide a platform for the tasks of dynamic 4D (3D+temporal) tracking, 5D (4D+interactive) interactive event recognition and intention prediction.

### Data

BLVD dataset contains 654 high-resolution video clips owing 120k frames extracted from Changshu, Jiangsu Province, China, where the Intelligent Vehicle Proving Center of China (IVPCC) is located. The frame rate is 10fps/sec for RGB data and 3D point cloud. We fully annotated all the frames and totally yield 249, 129 3D annotations, 4, 902 independent individuals for tracking with the length of overall 214, 922 points, 6, 004 valid fragments for 5D interactive event recognition, and 4, 900 individuals for 5D intention prediction. These tasks are contained in four kinds of scenarios depending on the object density (low and high) and light conditions (daytime and nighttime).

### Download
The 654 video clips (42.7GB) of BLVD can be downloaded [here](https://pan.baidu.com/s/1A6ggD7KOMZdlNmPMNO7KLg).and the password is **xpo4**.

### FILES
The dataset is saved in terms of the light conditions (daytime and nighttime) and the participant density (low and high). The file name is set as **“A-B-No.”**, where A denotes the light condition, such as **“D”** and **“N”**, and B specifies the participant density, such as  **“L”** and **“H”**, and No. is the clip index. For each clip, we save the original image data (each image has the resolution of 1920×500) and associated label file. The storage format of the label file can be seen in **label instruction** file.

### CITATION

```
@article{blvdICRA2019, 
  title={BLVD: Building A Large-scale 5D Semantics Benchmark for Autonomous Driving}, 
  author={Jianru Xue, Jianwu Fang, Tao Li, Bohua Zhang, Pu Zhang, Zhen Ye , Jian Dou}, 
  booktitle={Proc. International Conference on Robotics and Automation},, 
  year={2019}
}
```

### Videos

See
https://youtu.be/p6fSiPCg9Fs

