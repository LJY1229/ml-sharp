# 单场景：每点可训练特征 + 角度条件 MLP

本版本无需教师颜色，也不使用 fit_sharp_colors 的结果。直接将渲染图与 FLUX 目标图比较，训练每个 Gaussian 的32维特征和共享的3层128通道 MLP。几何、透明度、相机固定。直接预测 sigmoid RGB，不使用原色 logit 残差；初始渲染偏灰是正常现象。

这是单场景表达能力实验，每个模型只适用于自己的原始 Gaussian 场景。不能用训练好的每点特征直接处理新场景。没有显式遮挡计算，不能保证未训练角度的阴影物理正确。

## 安装与缓存

将包内整个 `sharp_relight` 目录放到 `/ml-sharp/`，用已有的 SHARP CUDA 环境。包中包含此前的公共工具，新旧入口互不覆盖；旧 MLP checkpoint 不能用于新模型。命令均在 `/ml-sharp` 运行。

数据仍采用你的原始结构：

```text
flux_relight_source/indoor/indoor_001/source.jpg
flux_relight_dataset/angle_000/indoor_001_angle_000.png
flux_relight_dataset/angle_030/indoor_001_angle_030.png
...
flux_relight_dataset/angle_180/indoor_001_angle_180.png
```

如果已有 `relight_cache_one/indoor_001.pt`，直接复用。不需要重新预测 SHARP。否则：

```bash
CUDA_VISIBLE_DEVICES=0 python sharp_relight/train_relight.py prepare \
  --scenes indoor_001 --cache relight_cache_one
```

可加 `--sharp-checkpoint /path/to/sharp_2572gikvuh.pt`；默认自动下载官方权重。

## 先跑短检查，再正式训练

```bash
CUDA_VISIBLE_DEVICES=0 python sharp_relight/train_point_latent.py train \
  --cache relight_cache_one --scene indoor_001 \
  --output runs/latent_smoke --steps 14 --log-every 1
```

确认没有梯度、CUDA或路径错误后，开启新的正式实验：

```bash
CUDA_VISIBLE_DEVICES=0 python sharp_relight/train_point_latent.py train \
  --cache relight_cache_one --scene indoor_001 \
  --output runs/latent_one --steps 7000 \
  --latent-dim 32 --hidden 128 --layers 3 \
  --feature-lr 0.01 --network-lr 0.001
```

默认7个角度均衡采样，每7步各出现一次、顺序打乱。7000步是总更新次数，每个角度约1000次，不是7000个epoch。步数增加和结构变化都可能影响效果；比较旧模型时也应给予相近的更新预算，不能把效果变化全部归因于结构。

训练直接预测 RGB，初始结果不再近似 SHARP 原色，所以初始损失可能更高。`preview_000000` 是初始网络输出；`baseline.png` 才是原始 SHARP 渲染。

## 查看结果与推理

- `preview_000500/angle_060.png` 等：原始 SHARP、FLUX目标、网络预测、绝对误差。
- 每个 preview 的 `metrics.json`：各角度 L1、PSNR及原始渲染 L1。
- `loss.csv`：更新前损失、每点特征梯度范数、MLP梯度范数。
- `last.pt`：每1000步及训练结束保存，包含每点特征、MLP和优化器状态。预览默认每500步保存。

```bash
CUDA_VISIBLE_DEVICES=0 python sharp_relight/train_point_latent.py infer \
  --cache relight_cache_one --model runs/latent_one/last.pt \
  --angle 60 --target-root flux_relight_dataset --output runs/latent_60

CUDA_VISIBLE_DEVICES=0 python sharp_relight/train_point_latent.py infer \
  --cache relight_cache_one --model runs/latent_one/last.pt \
  --angle 45 --output runs/latent_45
```

输出 `relit.png`、`relit.ply`、`colors_linear.pt`。指定 `--target-root` 且角度对应已有标签时额外输出对比图及误差。推理自动从模型读取场景名，并校验缓存文件哈希，防止把每点特征应用到点顺序不同的场景。

继续训练到总计14000步：

```bash
CUDA_VISIBLE_DEVICES=0 python sharp_relight/train_point_latent.py train \
  --cache relight_cache_one --scene indoor_001 --output runs/latent_one \
  --resume runs/latent_one/last.pt --steps 14000
```

恢复要求缓存、目标图、角度、max-side和seed不变；学习率使用已保存optimizer中的值。中断后从最近保存步恢复。

## 检验未见角度

创建独立训练运行，仅用 `--angles 0 60 120 180`，然后用 infer 对30、90、150度生成图像并加 `--target-root flux_relight_dataset` 比较。训练角度拟合好不代表中间角度正确，注意双重阴影、模糊及跳变。

## 显存与限制

默认单GPU。32维特征若对应118万个Gaussian，参数约151MB；加梯度和Adam两份状态约604MB，此外还有网络激活、渲染器及其他开销。checkpoint含optimizer，也会明显大于旧小MLP。使用分块和activation checkpointing，默认 `--chunk-size 32768`；显存不足可降到8192。`--max-side 512` 仅降低渲染分辨率，不减少Gaussian数量。无法在未实测前保证具体显存峰值或运行时长。

原图和目标必须对齐；允许同比例缩放，纵横比差异超过1%会拒绝。与旧脚本相同，线性RGB经SHARP渲染器转为sRGB后计算L1，固定黑色背景。导出PLY沿用SHARP官方颜色转换，第三方查看器可能与训练PNG存在差异。

本代码包带CPU测试：`python sharp_relight/test_point_latent.py`，检查特征和解码器的梯度、分块一致性、固定可微合成器下的学习、角度采样及模型恢复。真实SHARP/gsplat CUDA训练仍需在H100上验证。
