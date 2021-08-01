# DLPNet
# Dynamic Focal Loss with Reinforcement Learning for Aerial Imagery Detection

# Introduction
Object Detection on aerial imagery is a challenging task due to the following unique characteristics of the aerial imagery data: the large image size and the huge volume of data. Since the original image size from remote sensing sources is generally more colossal than the typical natural images, splitting an image into smaller image patches is a commonly used pre-processing method, where publicly available datasets are often distributed in smaller patches.
However, this process can result in a poor detection performance due to certain objects being located at the edge of image patches. Therefore, generally performance can be degraded, if one merely focuses on the small patch itself, instead of the original larger image.  
In addition, high computing resources are required to process the large amount of the original aerial images, where large image patches would need large data batch sizes. This can cause a significant bottleneck for the many practical real-world applications. 
Therefore, in this work, we propose a novel method to enable a robust and stable training with a small batch size as well. Our model composes of an oriented object detection model and a reinforcement learning agent. Our reinforcement learning agent extracts features from training batch data and determines optimal parameters to the focal loss function of the object detection model. This dynamic focal loss function with adaptive parameters can achieve a more robust and stable learning process than other advanced baseline model. We demonstrate the effectiveness of our approach with a well-known dataset, DOTA-v2.0.

<p align="center">
	<img src="imgs/diagram.png", width="800">
</p>

<p align="center">
	<img src="imgs/overview.png", width="800">
</p>

## Train Model
```ruby
!python main.py \
--num_epoch 50 \
--batch_size 8 \
--num_worker 8 \
--init_lr 5e-5 \
--input_h 600 \
--input_w 600 \
--K 1000 \
--conf_thresh 0.1 \
--ngpus 1 \
--dataset 'dota' \
--data_dir 'data_folder_path' \
--phase 'train' \
--seed 12345 \
--save_dir 'save_folder_path'
```

This implementation is extended from official BBAVectors code follow: [Code](https://github.com/yijingru/BBAVectors-Oriented-Object-Detection)

[Paper] Oriented Object Detection in Aerial Images with Box Boundary-Aware Vectors ([arXiv](https://arxiv.org/pdf/2008.07043.pdf))
	
	@inproceedings{yi2021oriented,
	title={Oriented object detection in aerial images with box boundary-aware vectors},
	author={Yi, Jingru and Wu, Pengxiang and Liu, Bo and Huang, Qiaoying and Qu, Hui and Metaxas, Dimitris},
	booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
	pages={2150--2159},
	year={2021}
	}
