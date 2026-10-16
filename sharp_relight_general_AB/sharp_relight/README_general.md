# 两个支持新图推理的重光照网络

这一版不使用场景ID、每点可训练特征或教师颜色。所有场景共享网络权重；新图直接经过冻结 SHARP 和训练好的重光照网络即可推理，无需新图的目标标签或梯度优化。支持推理不等于已经验证泛化效果：本机没有H100，也没有你的FLUX数据，实际效果须用留出场景检验。

## 两种结构

### A：复用冻结 SHARP 特征（`--architecture sharp`）

```text
原图 -> 冻结 SHARP -> texture_features + geometry_features
             |                        |
             -> Gaussian属性图 -------+ -> 1x1融合 -> 4个FiLM残差块 -> 颜色输出
原图RGB ------------------------------+                   ^
                                                       角度sin/cos
```

使用 `feature_model` 的 forward hook 提取两种预训练特征；使用 `init_model` hook 读取真实Gaussian层数和网格尺寸。无需修改官方源码。训练的只有新增融合层、角度编码和颜色头。保留Gaussian属性输入，便于不同层颜色预测。这个头本身不增加多尺度下采样，主要依靠SHARP已有特征提供上下文。

### B：从原图和Gaussian属性训练U-Net（`--architecture unet`）

```text
原图RGB + 各层Gaussian的RGB/XYZ/alpha
                 |
    32 -> 64 -> 128 -> 256 -> 128 -> 64 -> 32
            四尺度U-Net及跳跃连接
                 ^
        每个残差块接受角度FiLM调制
                 |
             颜色输出
```

B不读取SHARP的texture/geometry特征，仍使用冻结SHARP输出的几何。A和B默认直接预测线性RGB（sigmoid），不采用原色logit残差，减少暗色增亮受原色限制的问题。随机初始化结果可能偏灰，不是接口错误。输出对应每层Gaussian，展平后替换颜色；不是直接输出2D重光照图片。

两者均在 `general_models.py` 中定义，统一训练入口为 `general_relight.py`。模型参数量不同，因此比较不是严格的等参数量消融实验。

## 安装及目录

把压缩包中的 `sharp_relight` 文件夹放到 `/ml-sharp/`，使用已有SHARP CUDA环境。包中附带所需的旧工具模块，不要只复制两个新文件。若包尚不能导入，在仓库根目录执行 `pip install -e . --no-deps`。命令均在 `/ml-sharp` 执行。

自动匹配你的目录：

```text
flux_relight_source/indoor/indoor_001/source.jpg
flux_relight_source/outdoor/outdoor_050/source.jpg
flux_relight_dataset/angle_000/indoor_001_angle_000.png
flux_relight_dataset/angle_030/indoor_001_angle_030.png
...
```

## 1. 共用的特征缓存

旧版Gaussian缓存没有SHARP特征和网格元数据，需要新建缓存：

```bash
CUDA_VISIBLE_DEVICES=0 python sharp_relight/general_relight.py prepare \
  --cache relight_feature_cache --feature-side 384
```

提取100个原图场景，冻结模型，缓存linearRGB Gaussian、纹理/几何特征、原图RGB、网格尺寸与相机信息。特征默认缩小至最长边384，以float16存储，训练转换回float32；这会丢失一部分细节，换来的好处是缓存更小。可用 `--feature-side 768` 保留更高分辨率，但必须使用新缓存目录。

为了两个版本共用一个缓存，准备步骤都提取SHARP特征；B训练不会将这些特征送入网络。缓存约包括每场景Gaussian的 N×14×4 字节，以及 C×H×W×2 字节的特征；若C=64、特征384方形，仅特征约19MB/场景。

可选 `--sharp-checkpoint /path/to/sharp_2572gikvuh.pt`；不指定则使用官方默认下载。新图推理需使用与prepare相同的权重选项。

默认按场景分层划分80训练、10验证、10测试（indoor/outdoor各自划分）。记录在缓存 `manifest.json` 中，两版本完全共用划分。同一场景的七个角度不会跨集合。

## 2. A版训练

```bash
CUDA_VISIBLE_DEVICES=0 python sharp_relight/general_relight.py train \
  --architecture sharp --cache relight_feature_cache \
  --output runs/sharp_feature_head --epochs 30
```

## 3. B版训练

```bash
CUDA_VISIBLE_DEVICES=1 python sharp_relight/general_relight.py train \
  --architecture unet --cache relight_feature_cache \
  --output runs/unet_relight --epochs 30
```

准备完缓存后可两卡各跑一个实验。每个脚本是单卡训练，不支持torchrun/DDP。训练默认float32、Adam学习率3e-4、每步一个场景一个角度、模型工作网格384×384、渲染最长边512。每个epoch遍历训练集所有角度，默认560次更新。只优化新网络，SHARP不参与反向传播。每5个epoch评估验证集，以其平均L1保存 `best.pt`；每epoch保存可恢复的 `last.pt`。

训练快照：`val_0005/indoor_XXX_060.png`，从左到右是原始SHARP渲染、目标、预测、绝对误差。损失为sRGB渲染图与FLUX目标的L1；可同时查看PSNR和baseline_l1。两版采用相同损失和默认更新预算。

断点恢复例如：

```bash
CUDA_VISIBLE_DEVICES=0 python sharp_relight/general_relight.py train \
  --architecture sharp --cache relight_feature_cache \
  --output runs/sharp_feature_head --resume runs/sharp_feature_head/last.pt --epochs 60
```

epochs是总epoch数；恢复时保持数据、角度、seed、model-side、max-side等设置不变，优化器学习率以checkpoint为准。勿在恢复时修改缓存和标签内容。

## 4. 输入一张新图片和角度

无需先给新图制作数据集，不需要FLUX标签，也不需要单场景优化：

```bash
CUDA_VISIBLE_DEVICES=0 python sharp_relight/general_relight.py infer \
  --model runs/sharp_feature_head/best.pt \
  --source /path/to/new_image.jpg --angle 45 --output runs/new_image_sharp_45

CUDA_VISIBLE_DEVICES=1 python sharp_relight/general_relight.py infer \
  --model runs/unet_relight/best.pt \
  --source /path/to/new_image.jpg --angle 45 --output runs/new_image_unet_45
```

推理依次加载冻结SHARP和重光照网络，输出 `baseline.png`、`relit.png`、`relit.ply`、`colors_linear.pt`。额外保留 `scene_cache/input.pt`，可用于后续扩展交互式多角度推理。当前CLI每次infer会重新运行SHARP，不是已经优化好的实时交互程序。

导出PLY是这一角度下的静态颜色，改变光照需再次调用网络。第三方PLY查看器的颜色处理可能与SHARP渲染器不同。

## 5. 比较新场景泛化

```bash
CUDA_VISIBLE_DEVICES=0 python sharp_relight/general_relight.py evaluate \
  --model runs/sharp_feature_head/best.pt --cache relight_feature_cache \
  --split test --output runs/test_sharp

CUDA_VISIBLE_DEVICES=1 python sharp_relight/general_relight.py evaluate \
  --model runs/unet_relight/best.pt --cache relight_feature_cache \
  --split test --output runs/test_unet
```

evaluate会从模型恢复model-side，默认max-side=512。若训练渲染尺寸不是512，评估时明确指定一致的max-side。测试集用于最终评估，不能反复用测试结果选模型。想检验角度插值，可另开实验训练 `--angles 0 60 120 180`，评估 `--angles 30 90 150`，与全角度训练实验分开。

## 先做小实验（建议）

可在完整缓存上直接添加 `--scenes indoor_001`：

```bash
CUDA_VISIBLE_DEVICES=0 python sharp_relight/general_relight.py train \
  --architecture sharp --cache relight_feature_cache --scenes indoor_001 \
  --output runs/smoke_sharp --epochs 1 --log-every 1

CUDA_VISIBLE_DEVICES=1 python sharp_relight/general_relight.py train \
  --architecture unet --cache relight_feature_cache --scenes indoor_001 \
  --output runs/smoke_unet --epochs 1 --log-every 1
```

若未准备完整缓存，也可先prepare加 `--scenes indoor_001`，用独立cache目录。小实验通过后可各训练100～300个epoch观察单场景拟合，再进入多场景实验。显式指定scenes时关闭验证集，输出 `train_preview_*`，只有 `last.pt`，推理相应使用last。单场景训练出的网络虽然接口支持新图，不能因此声称已经学会泛化。

## 分辨率与限制

- Gaussian真实网格来自SHARP initializer hook，并验证N=L×H×W。严格按照层、高、宽展平，避免混层错位。
- 模型在384网格预测，再上采样到原Gaussian网格。因此比逐点自由优化多了空间带宽限制，细阴影边缘可能模糊。如果原fit_sharp_colors好而这里边缘差，可在显存允许时将两版都提高到 `--model-side 768`，A最好也准备768特征。两种分辨率必须保持可比较。
- U-Net输入包括各层RGB、归一化XYZ、alpha与原图，默认2层为17通道。A额外拼接预训练特征，通道数从缓存读取，不写死。
- 原图/目标必须内容对齐，允许同比例不同分辨率；纵横比差异超过1%会拒绝。
- 原图缺焦距时沿用SHARP默认估计，单位外参、缩放后的原相机内参用于训练。
- 两版都没有显式光线遮挡计算，SHARP预训练特征也不等于材质或可见性。FLUX结构变化会干扰训练；100场景下新图的阴影准确性仍需实验验证。
- 当前代码对照SHARP commit aed6527499ef91cba3b54c18d49a870f25947190。本机仅CPU，不能执行实际SHARP/gsplat CUDA训练。

CPU测试命令：`python sharp_relight/test_general.py`。覆盖层/网格映射、两版的参数梯度和模型恢复，以及通过场景/角度条件学习合成颜色目标；不能替代H100端到端测试。
