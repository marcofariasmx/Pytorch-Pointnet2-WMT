# Pytorch Implementation of PointNet and PointNet++ 

This repo is implementation for [PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) and [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf) in pytorch.

The original repo is found [here](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).

Research paper:
https://arxiv.org/pdf/1612.00593.pdf

## Updates
**2023/09/21:** 

(1) Fixed some bugs and improved overall readability and updated to Pytorch 2.0 and Python 3.10.

**2023/07/10:**

(1) Forked original PointNet++ version and adapted it to suit the WMT's data structures and needs.

## Install
The latest codes are tested both on Ubuntu 18.04 running on a VM on WSL and Windows 11. CUDA11.7, PyTorch 2.0 and Python 3.10:

For proper installation based on your system, please visit and follow the steps on [installing Pytorch](https://pytorch.org/get-started/locally/)

## Important note
Multithreading (which greatly improves peformance) does not work for Windows unless the program is run on an Ubuntu VM.

## Classification
Disabled at the moment.

## Part Segmentation
### Data Preparation
### Run
```
## Check for different model in ./models 
## e.g., pointnet2_msg
python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg --num_workers 1
python test_partseg.py --normal --log_dir pointnet2_part_seg_msg --num_workers 1
```


## Semantic Segmentation (S3DIS)
### Data Preparation
Follow the steps on the Semantic Segmentation - Rack Extraction Jupyter notebook to convert pointcloud data to text files and place it under the `/data/s3dis/Dataset` directory. 


Run
```
python create_files.py
```
The previous line creates the needed file content for the `./data_utils/meta/anno_paths.txt` and 
`/data_utils/meta/class_names.txt` files based on the content on the `/data/s3dis/Dataset` directory.

#### Make sure that the classes created in `/data_utils/meta/class_names.txt` match with the classes found in the 
`g_class2color` dictionary found in `/data_utils/indor3d_util.py`

The following command proceeds to preprocess the data by loading the point cloud labeled text files and store them into
numpy arrays.
```
python data_utils/collect_indoor3d_data.py
```
Processed data will be saved in `data/processed_data/`.
### Run
```
## Check model in ./models 
## e.g., pointnet2_ssg
python train_semseg.py --model pointnet2_sem_seg --test_area 1 --log_dir pointnet2_sem_seg --gpu 0
python test_semseg.py --log_dir pointnet2_sem_seg --test_area 1 --visual
```
Visualization results will save in `log/sem_seg/pointnet2_sem_seg/visual/` and you can visualize these .obj file by [MeshLab](http://www.meshlab.net/).

### Note:
Even though it is not mentioned in the test examples, the msg (Multi-Scale Grouping) model can be also used for semantic segmentation.
```
python train_semseg.py --model pointnet2_sem_seg_msg --test_area 1 --log_dir pointnet2_sem_seg_msg --gpu 1
```


## Visualization
### Using of CloudCompare https://www.danielgm.net/cc/

### Using MeshLab
![](/visualizer/pic2.png)


## Reference By
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)<br>
[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)<br>
[charlesq34/PointNet](https://github.com/charlesq34/pointnet) <br>
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)
