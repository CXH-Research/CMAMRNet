# 📋 Mural Restoration

[CMAMRNet: A Contextual Mask-Aware Network Enhancing Mural Restoration Through Comprehensive Mask Guidance](https://arxiv.org/abs/2508.07140)

[Yingtie Lei](https://github.com/yingtie-lei), Fanghai Yi, Yihang Dong, Weihuang Liu, [Xiaofeng Zhang](https://zhangbaijin.github.io), Zimeng Li, [Chi-Man Pun](https://www.cis.um.edu.mo/~cmpun/) and [Xuhang Chen](https://cxh.netlify.app/) 📮 (📮 Corresponding authors)

**University of Macau, Guangdong University of Technology, University of Chinese Academy of Sciences, Shanghai Jiao Tong University, Shenzhen Polytechnic University, Huizhou University**

2025 British Machine Vision Conference (BMVC 2025)

# 🔮 Dataset

The dataset is available at [Kaggle](https://www.kaggle.com/datasets/xuhangc/dunhuang-grottoes-painting-dataset-and-benchmark).

# ⚙️ Usage

## Training

You may download the dataset first, and then specify TRAIN_DIR, VAL_DIR and SAVE_DIR in the section TRAINING in `config.yml`.

For single GPU training:

```bash
python train.py
```

For multiple GPUs training:

```bash
accelerate config
accelerate launch train.py
```

If you have difficulties with the usage of `accelerate`, please refer to [Accelerate](https://github.com/huggingface/accelerate).

## Inference

```bash
python test.py
```

# 💗 Acknowledgement

This work was supported in part by the Science and Technology Development Fund, Macau SAR, under  Grant 0141/2023/RIA2 and 0193/2023/RIA3, and the University of Macau under Grant MYRG-GRG2024-00065-FST-UMDF, in part by the Shenzhen Polytechnic University Research Fund (Grant No. 6025310023K).

# 🛎 Citation

If you find our work helpful for your research, please cite:

```bib
```
