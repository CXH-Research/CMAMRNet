# 📋 Mural Restoration

CMAMRNet: A Contextual Mask-Aware Network Enhancing Mural Restoration Through Comprehensive Mask Guidance

[Yingtie Lei](https://github.com/yingtie-lei), Fanghai Yi, Yihang Dong, Weihuang Liu, [Xiaofeng Zhang](https://zhangbaijin.github.io), Zimeng Li, [Chi-Man Pun](https://www.cis.um.edu.mo/~cmpun/) and [Xuhang Chen](https://cxh.netlify.app/) 📮 (📮 Corresponding authors)

**University of Macau, Guangdong University of Technology, Shanghai Jiao Tong University, Shenzhen Polytechnic University, Huizhou University**

2025 British Machine Vision Conference (BMVC 2025)

# 🔮 Dataset

The dataset is available at Kaggle.

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

This work was supported in part by the Science and Technology Development Fund, Macau SAR, under Grant 0141/2023/RIA2 and 0193/2023/RIA3, in part by the National Key Research and Development Program of China (No. 2023YFC2506902), the National Natural Science Foundations of China under Grant 62172403, the Distinguished Young Scholars Fund of Guangdong under Grant 2021B1515020019.

# 🛎 Citation

If you find our work helpful for your research, please cite:

```bib
```