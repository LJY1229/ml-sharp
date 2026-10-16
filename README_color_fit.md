# SHARP 固定几何颜色拟合实验

将 `fit_sharp_colors.py` 放到已安装依赖的 ml-sharp 仓库根目录，在你的 H100/CUDA 环境运行。此脚本复用 `sharp.cli.predict.predict_image`、`GSplatRenderer`、`save_ply`，不修改 SHARP 源码。

## 数据

```text
data/test_scene/
  source.png
  targets/
    000.png
    030.png
    060.png
    090.png
    120.png
    150.png
    180.png
```

source 是原图，targets 是同一相机、同一场景的重光照目标。目标图必须与原图尺寸相同、内容对齐。脚本检查尺寸，但无法自动保证内容对齐。不要将 source 放进 targets 目录，除非有意将其作为自重建测试。文件名只用于标记结果，不作为角度输入。

## 运行

已按 SHARP README 安装环境后，在仓库根目录确保包可导入（必要时 `pip install -e . --no-deps`）。Linux H100 服务器示例：

```bash
# 推荐首先拟合原图本身，检查相机及优化链路。
CUDA_VISIBLE_DEVICES=0 python fit_sharp_colors.py --source data/test_scene/source.png --targets data/test_scene/source.png --output runs/self_fit --steps 500

# 再试一张有明显阴影的目标图。
CUDA_VISIBLE_DEVICES=0 python fit_sharp_colors.py --source data/test_scene/source.png --targets data/test_scene/targets/060.png --output runs/fit_060 --steps 1000 --lr 0.01 --max-side 512

# 最后独立拟合七张目标。只预测一次 SHARP 场景，各目标颜色独立初始化。
CUDA_VISIBLE_DEVICES=0 python fit_sharp_colors.py --source data/test_scene/source.png --targets data/test_scene/targets --output runs/fit_all --steps 1000 --lr 0.01 --max-side 512
```

可加 `--checkpoint /path/to/sharp_2572gikvuh.pt` 使用本地权重，否则使用 SHARP 官方默认下载地址。没有 EXIF 焦距时，沿用 SHARP 的估计；如果知道原图像素单位的焦距，可用 `--focal-px` 指定。输出目录必须为新目录或空目录。

`--max-side` 只降低渲染/监督分辨率，不改变 SHARP 的 1536 内部推理分辨率，也不减少 Gaussian 数量。所有操作使用 float32。脚本要求 CUDA，不支持 CPU 渲染训练。

## 输出

- `source.png`、`baseline.png`、`baseline_error.png`：缩放后的原图、初始渲染和两者绝对误差，用来排查相机与几何。
- `base_gaussians.pt`：原始 Gaussian 属性字典，CPU tensor。
- `config.json`：相机焦距、分辨率、颜色空间及实验设置。
- 各目标子目录下 `comparison.png`：从左到右为初始渲染、目标图、优化结果、绝对误差（无放大）。
- `step_*.png`、`loss.csv`：训练过程；CSV 损失在当前更新之前，快照在更新之后。
- `colors_linear.pt`：优化后的线性 RGB，顺序对应 base_gaussians.pt。
- `fitted.ply`：调用 SHARP 官方导出函数生成的场景，可用于查看小幅转动视角后的表现。
- `metrics.json`：初始/最终的 sRGB L1 和 PSNR，以及初始渲染对原图的误差。

官方 PLY 导出涉及线性 RGB 到 sRGB 的转换与文件存储精度，第三方查看器的结果也受其颜色处理影响。精确实验结果以本脚本 PNG 和保存的线性颜色 tensor 为准。

## 实验解释

唯一优化变量是 Gaussian RGB；位置、尺度、旋转、透明度、相机和黑色背景全部固定。优化在线性 RGB 参数上进行，渲染器转为 sRGB 后与目标计算 L1。使用投影式 Adam：每次更新后将 RGB 限制到 [0,1]。没有材质分解、光照模型、角度条件网络或阴影监督。

先看原图自拟合能否收敛。重光照目标拟合成功，只说明固定几何上的颜色自由度足以在该视角表示目标亮暗；不说明学到了真实光照、角度插值或新场景泛化。拟合失败也不是数学上的不可能证明，应检查迭代次数、学习率、对齐、alpha 覆盖、Gaussian 大小和目标结构变化。若出现振荡可降学习率到 0.003；512 下通过后可增大渲染分辨率检验阴影边界。

## 验证状态

已对照当前拉取的 SHARP 源码核对接口并进行 Python 语法检查。本机 PyTorch 是 CPU 版本，未执行真实 SHARP 权重推理和 gsplat CUDA 前向/反向；须在你的 H100 环境运行验证。脚本第一步会检查颜色梯度是否存在、有限且非零。
