# SHARP + 角度条件 MLP：最小可训练重光照基线

这是一个新增神经网络的训练实现。冻结 SHARP，缓存原图 Gaussian；训练共享的 MLP 权重，通过 Gaussian 可微渲染后的图像与 FLUX 目标图计算 L1。不是直接优化每个点的颜色参数。

## 文件与安装

将整个 `sharp_relight` 文件夹放到 `/ml-sharp/`，使用已有的 SHARP CUDA 环境。无需改 SHARP 源码，也无需增加第三方依赖。若仓库源码尚不可导入，先在仓库根目录执行 `pip install -e . --no-deps`。本代码对照 apple-aiml-research/ml-sharp commit `aed6527499ef91cba3b54c18d49a870f25947190` 的接口编写。

```text
/ml-sharp/
  sharp_relight/
    relight_mlp.py       # 新增网络
    train_relight.py     # 缓存、训练、评估、推理
    test_relight.py      # CPU 单元测试
  flux_relight_source/
    indoor/indoor_001/source.jpg
    ...
    outdoor/outdoor_050/source.jpg
  flux_relight_dataset/
    angle_000/indoor_001_angle_000.png
    angle_000/outdoor_050_angle_000.png
    angle_030/indoor_001_angle_030.png
    ...
    angle_180/outdoor_050_angle_180.png
```

以下命令均在 `/ml-sharp` 执行。默认用单张 H100，不支持 torchrun/DDP；不要用两进程同时写同一 cache/run。你可以将另一张卡用于不同输出目录的独立实验。

## 1. 先做一个场景的实验

先缓存一个场景，避免首次就跑全部数据：

```bash
CUDA_VISIBLE_DEVICES=0 python sharp_relight/train_relight.py prepare \
  --scenes indoor_001 --cache relight_cache_one
```

会自动下载 SHARP 权重；也可加 `--sharp-checkpoint /path/to/sharp_2572gikvuh.pt`。缓存内保存所有 Gaussian 的 float32 参数及原始相机焦距、分辨率。中断后原命令可复用已完成的场景。

训练全部七个角度，先看能否拟合（100 epochs = 700 次参数更新，不是 70000 次）：

```bash
CUDA_VISIBLE_DEVICES=0 python sharp_relight/train_relight.py train \
  --cache relight_cache_one --scenes indoor_001 \
  --output runs/mlp_one --epochs 100 --eval-every 10 --log-every 7
```

建议先 `--epochs 1 --log-every 1`，确认 CUDA 编译、梯度和输出正常；使用独立输出目录，如 `runs/smoke`。

查看 `runs/mlp_one/train_preview_0100/indoor_001_060.png`：从左到右是初始渲染、目标图、网络预测、绝对误差。这里是训练样本预览，不能称为验证结果。`last.pt` 是训练出的网络，`loss.csv` 记录每个样本更新前的损失。单场景模式没有 `best.pt`，使用 `last.pt`。

输入一个连续角度并导出 3DGS：

```bash
CUDA_VISIBLE_DEVICES=0 python sharp_relight/train_relight.py infer \
  --cache relight_cache_one --model runs/mlp_one/last.pt \
  --scene indoor_001 --angle 45 --output runs/infer_one_45
```

输出 `relit.png`、`baseline.png`、`relit.ply`、`colors_linear.pt`。45°没有目标图，这里只能视觉观察；连续角度的准确性不保证。

## 2. 独立测试角度插值

仅训练四个方向，其他三个方向完全不参与参数更新：

```bash
CUDA_VISIBLE_DEVICES=0 python sharp_relight/train_relight.py train \
  --cache relight_cache_one --scenes indoor_001 --angles 0 60 120 180 \
  --output runs/mlp_angle_holdout --epochs 100 --eval-every 10

CUDA_VISIBLE_DEVICES=0 python sharp_relight/train_relight.py evaluate \
  --cache relight_cache_one --model runs/mlp_angle_holdout/last.pt \
  --scenes indoor_001 --angles 30 90 150 --output runs/eval_angle_holdout
```

这测试同一场景的未见角度，不是未见场景泛化。对七个方向都训练过的模型再评估这三个方向，不能称为角度留出实验。

## 3. 全数据训练与新场景评估

```bash
CUDA_VISIBLE_DEVICES=0 python sharp_relight/train_relight.py prepare --cache relight_cache

CUDA_VISIBLE_DEVICES=0 python sharp_relight/train_relight.py train \
  --cache relight_cache --output runs/mlp_all --epochs 30 --eval-every 5

CUDA_VISIBLE_DEVICES=0 python sharp_relight/train_relight.py evaluate \
  --cache relight_cache --model runs/mlp_all/best.pt \
  --split test --output runs/mlp_test
```

100 场景齐全时，默认按 indoor/outdoor 分层、固定随机种子划分 80/10/10 个训练/验证/测试场景。同一场景的所有角度保持在同一个集合。划分记录在 cache 的 `manifest.json` 中；测试集不参与训练和 best checkpoint 选择。小于每类10个场景时该类可能没有验证/测试样本。

每个 epoch 遍历全部训练场景和选定角度；先打乱场景，再打乱该场景的角度，以避免每个角度反复加载大场景缓存。每步一个场景、一个角度。全部数据默认每个 epoch 有 80×7=560 次更新。

继续训练（`--epochs` 是总 epoch 数，恢复时学习率以保存的 optimizer 为准）：

```bash
CUDA_VISIBLE_DEVICES=0 python sharp_relight/train_relight.py train \
  --cache relight_cache --output runs/mlp_all --epochs 60 \
  --resume runs/mlp_all/last.pt
```

恢复时需保持场景、角度、seed、max-side 与原训练一致。checkpoint 按 epoch 保存，中断的 epoch 会重新执行。完整运行配置保存在 `config.json`；不要在恢复时替换目标数据内容。

## 网络到底在训练什么

每个 Gaussian 的输入：归一化 XYZ（3维）+ XYZ 的4级 Fourier编码（24维）+ 原始线性RGB（3维）+ 光照角度 sin/cos（2维），共32维。3层64通道 SiLU MLP，再输出3维颜色 logit 残差。约1.1万参数，默认输出头初始化为零。

```text
fixed source image -> frozen SHARP -> fixed Gaussian geometry / original RGB
                                         |
normalized XYZ + original RGB + angle -> shared RelightMLP
                                         |
                               predicted linear RGB
                                         |
                      differentiable Gaussian renderer
                                         |
                           sRGB render vs target: L1
                                         |
                           backward updates MLP only
```

`sigmoid(logit(original_rgb.clamp(1e-4,1-1e-4)) + delta)` 保持颜色在0到1，初始化接近原图外观。原始深暗颜色的增亮可能较慢，这是这一简单参数化的局限。加入低阶 XYZ Fourier 编码，帮助小 MLP 表达空间变化。没有可训练的场景ID或每点embedding，因此模型可以应用于缓存中的未见场景。

这仍是逐点共享 MLP，不读取其他点的邻域或场景上下文，也没有显式光源可见性、法线、材质和物理阴影计算。它能学习颜色变化规律，但不保证跨场景遮挡推理；多场景效果差不能单独否定整个重光照想法。不要把这份最小基线当成已经解决物理重光照的模型。

## 相机、颜色与数据要求

- 使用原相机的单位外参；内参沿用 SHARP render CLI 的像素中心约定，按渲染分辨率缩放。原图缺少焦距 EXIF 时沿用 SHARP 估计。
- 原图和目标必须内容对齐。允许相同比例的不同分辨率，统一缩放到训练分辨率；纵横比相差超过1%会拒绝，避免默默拉伸。
- SHARP Gaussian 是 linearRGB；用 `GSplatRenderer(color_space="linearRGB")` 转为 sRGB 后，与PNG目标做 L1。固定黑色背景，alpha不训练。
- `--max-side 512` 只减少渲染/监督尺寸，不减少 SHARP 原本的 Gaussian 数量。高分辨率请在小实验通过后再做。
- 原始PLY导出使用官方 `save_ply`，涉及 sRGB 转换；第三方查看器可能显示差异。精确实验数据以渲染PNG与线性RGB tensor为准。
- 新场景也可以放入相同 source 文件夹结构，用另一个 cache 执行 prepare，再使用已有模型 infer；infer 不需要目标图。

## 显存和检查

默认 float32，不训练 SHARP、不在训练阶段加载它的权重。MLP 对 Gaussian 分块，并通过 activation checkpointing 在反向时重算，减少中间激活显存；所有 Gaussian 仍参与渲染。显存不足可将 `--chunk-size 65536` 降为16384，或降低 `--max-side`。缓存每个场景约为 N×14×4 字节，加少量元数据；若 N≈118万，则约66MB/场景，100场景约6.6GB（十进制估算）。缓存不存整套原网络特征。

CPU检查：

```bash
python sharp_relight/test_relight.py
```

4项CPU测试已通过，命令行帮助已检查。验证范围：数据映射和场景划分、零残差初始化、分块反向梯度与完整前向一致，以及通过一个固定可微合成器学会不同角度。当前编写环境是 CPU PyTorch，尚未执行真实 SHARP/gsplat CUDA 集成训练；不能将CPU测试通过等同于H100端到端验证。训练首次更新会检查网络梯度，非有限损失/梯度会停止并报告。
