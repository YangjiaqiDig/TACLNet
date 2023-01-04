# A Topological-Attention ConvLSTM Network and Its Application to EM Images

This is an implementation of the [A Topological-Attention ConvLSTM Network and Its Application to EM Images](https://arxiv.org/abs/2202.03430).

## Abstract
Structural accuracy of segmentation is important for fine-scale structures in biomedical images. We propose a novel Topological-Attention ConvLSTM Network (TACLNet) for 3D anisotropic image segmentation with high structural accuracy. We adopt ConvLSTM to leverage contextual information from adjacent slices while achieving high efficiency. We propose a Spatial Topological-Attention (STA) module to effectively transfer topologically critical information across slices. Furthermore, we propose an Iterative Topological-Attention (ITA) module that provides a more stable topologically critical map for segmentation. Quantitative and qualitative results show that our proposed method outperforms various baselines in terms of topology-aware evaluation metrics.

## Data
#### Download [CREMI, ISBI12, and ISBI13](https://drive.google.com/drive/folders/1x9eeyZGUEiBSiDt8ZL1hzrGSHtM7aSfD?usp=share_link)

## Citation
Please cite our paper if the code is helpful to your research.
```
@inproceedings{yang2021topological,
  title={A topological-attention ConvLSTM network and its application to EM images},
  author={Yang, Jiaqi and Hu, Xiaoling and Chen, Chao and Tsai, Chialing},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={217--228},
  year={2021},
  organization={Springer}
}
```
