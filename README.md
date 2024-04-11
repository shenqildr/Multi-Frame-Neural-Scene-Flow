# Multi-Frame-Neural-Scene-Flow

[Dongrui Liu](https://shenqildr.github.io/), [Xueqian Li](https://lilac-lee.github.io/), and [Lei Chu](https://wides.usc.edu/#people)

arXiv link: [https://arxiv.org/pdf/2304.09121.pdf](https://arxiv.org/pdf/2403.16116v1.pdf)

---

Neural Scene Flow Prior (NSFP) and Fast Neural Scene Flow (FNSF) have shown remarkable adaptability in the context of large out-of-distribution autonomous driving. Despite their success, the underlying reasons for their astonishing generalization capabilities remain unclear. Our research addresses this gap by examining the generalization capabilities of NSFP through the lens of uniform stability, revealing that its performance is inversely proportional to the number of input point clouds. This finding sheds light on NSFP’s effectiveness in handling large-scale point cloud scene flow estimation tasks. Motivated by such theoretical insights, we further explore the improvement of scene flow estimation by leveraging historical point clouds across multiple frames, which inherently increases the number of point clouds. Consequently, we propose a simple and effective method for multi-frame point cloud scene flow estimation, along with a theoretical evaluation of its generalization abilities. Our analysis confirms that the proposed method maintains a limited generalization error, suggesting that adding multiple frames to the scene flow optimization process does not detract from its generalizability. Extensive experimental results on large-scale autonomous driving Waymo Open and Argoverse lidar datasets demonstrate that the proposed method achieves state-of-the-art performance.

---

### Prerequisites
This code is based on PyTorch implementation, and tested on PyTorch=1.13.0, Python=3.10.8 with CUDA 11.6 or PyTorch=1.12.0, Python=3.9.15 with CUDA 11.6. 
But it should work fine with a higher version of PyTorch.

A simple installation is ```bash ./install.sh```. For a detailed installation guide, please refer to [FastNSF](https://github.com/Lilac-Lee/FastNSF).


### Dataset
We provide datasets we used in our paper.
You may download datasets used in the paper from these links:

- [Argoverse](https://drive.google.com/drive/folders/1xxHJq0OtpR_aH-k6PvR0g4CkI3z_jUa2?usp=sharing)

- [Waymo Open](https://drive.google.com/drive/folders/1eS0QV_2afEWi89NF_l1oB7Vb0amtMUy9?usp=sharing)

After you download the dataset, you can create a symbolic link in the ./dataset folder as ```./dataset/argoverse``` and ```./dataset/waymo```.
<br></br>

---
### Optimization

- Waymo Open scene flow dataset
    ```
    python optimization.py --device cuda:0 --dataset WaymoOpenSceneFlowDataset --dataset_path ./dataset/waymo --exp_name opt_waymo_open_full_points --batch_size 1 --use_all_points --iters 5000 --model neural_prior --hidden_units 128 --layer_size 8 --lr 0.01 --act_fn relu --earlystopping --early_patience 10 --early_min_delta 0.001 --grid_factor 10 --init_weight --layer_size_align 3
    ```

- Argoverse scene flow dataset

    ```
    python optimization.py --device cuda:0 --dataset ArgoverseSceneFlowDatase --dataset_path ./dataset/argoverse --exp_name opt_waymo_open_full_points --batch_size 1 --use_all_points --iters 5000 --model neural_prior --hidden_units 128 --layer_size 8 --lr 0.01 --act_fn relu --earlystopping --early_patience 10 --early_min_delta 0.001 --grid_factor 10 --init_weight --layer_size_align 3
    ```



---

### Acknowledgement
[Neural Scene Flow Prior](https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior)

[Fast Neural Scene Flow](https://github.com/Lilac-Lee/FastNSF))

[FastGeodis: Fast Generalised Geodesic Distance Transform](https://github.com/masadcv/FastGeodis)

### Contributing
If you find the project useful for your research, you may cite,
```
@article{li2021neural,
  title={Neural Scene Flow Prior},
  author={Li, Xueqian and Kaesemodel Pontes, Jhony and Lucey, Simon},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}

@InProceedings{Li_2023_ICCV,
  title={Fast Neural Scene Flow},
  author={Li, Xueqian and Zheng, Jianqiao and Ferroni, Francesco and Pontes, Jhony Kaesemodel and Lucey, Simon},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month={October},
  year={2023},
  pages={9878-9890}
}

@article{liu2024self,
  title={Self-Supervised Multi-Frame Neural Scene Flow},
  author={Liu, Dongrui and Liu, Daqi and Li, Xueqian and Lin, Sihao and Wang, Bing and Chang, Xiaojun and Chu, Lei and others},
  journal={arXiv preprint arXiv:2403.16116},
  year={2024}
}
```
