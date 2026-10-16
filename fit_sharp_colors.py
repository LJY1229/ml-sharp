"""Fit only SHARP Gaussian RGB to aligned relighting images (CUDA required)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from sharp.cli.predict import DEFAULT_MODEL_URL, predict_image
from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import Gaussians3D, save_ply
from sharp.utils.gsplat import GSplatRenderer


def save_image(tensor, path):
    array = tensor.detach()[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    Image.fromarray(np.round(array * 255).astype(np.uint8)).save(path)


def image_tensor(array, device, size):
    value = torch.from_numpy(array.copy()).to(device=device, dtype=torch.float32)
    value = value.permute(2, 0, 1).unsqueeze(0) / 255
    return F.interpolate(value, size=size, mode="bilinear", align_corners=False,
                         antialias=True)


def metrics(pred, target):
    mse = F.mse_loss(pred, target).item()
    return {"l1": F.l1_loss(pred, target).item(),
            "psnr_db": float(-10 * np.log10(max(mse, 1e-12)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True, nargs="+",
                        help="One or more image files or directories of aligned PNG/JPEG images")
    parser.add_argument("--output", type=Path, required=True,
                        help="New/empty output directory (prevents accidental overwrite)")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max-side", type=int, default=512,
                        help="Training render size; does not reduce Gaussian count")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--focal-px", type=float,
                        help="Optional focal length in original source-image pixels")
    args = parser.parse_args()
    if min(args.steps, args.max_side, args.save_every) <= 0 or args.lr <= 0:
        parser.error("steps, max-side, save-every and lr must be positive")
    if args.focal_px is not None and args.focal_px <= 0:
        parser.error("focal-px must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run in the SHARP environment on your H100.")
    if not args.source.is_file() or not all(t.exists() for t in args.targets):
        parser.error("source and all targets must exist")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be a new or empty directory")

    extensions = {".png", ".jpg", ".jpeg"}
    targets = []
    for t in args.targets:
        if t.is_file():
            targets.append(t)
        else:
            targets.extend(sorted(p for p in t.iterdir() if p.suffix.lower() in extensions))
    if not targets:
        parser.error("No target images found")
    source, _, focal = io.load_rgb(args.source)
    h, w = source.shape[:2]
    # Targets may differ in resolution; image_tensor resizes all to (rh, rw).
    focal = args.focal_px if args.focal_px is not None else focal
    ratio = min(1., args.max_side / max(h, w))
    rh, rw = max(1, round(h * ratio)), max(1, round(w * ratio))
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config[key] = str(value)
        elif isinstance(value, list):
            config[key] = [str(v) if isinstance(v, Path) else v for v in value]
        else:
            config[key] = value
    config.update(focal_px_used=focal, source_hw=[h, w], render_hw=[rh, rw],
                  gaussian_color_space="linearRGB", loss_color_space="sRGB")
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("Loading SHARP and predicting the source scene...", flush=True)
    state = (torch.load(args.checkpoint, map_location="cpu", weights_only=True)
             if args.checkpoint else torch.hub.load_state_dict_from_url(
                 DEFAULT_MODEL_URL, map_location="cpu", progress=True))
    model = create_predictor(PredictorParams())
    model.load_state_dict(state)
    model.eval().requires_grad_(False).to(device)
    base = predict_image(model, source, focal, device)
    base = Gaussians3D(*(value.detach() for value in base))
    del model, state
    torch.cuda.empty_cache()
    torch.save({name: value.cpu() for name, value in base._asdict().items()},
               args.output / "base_gaussians.pt")

    # Same original camera convention as SHARP's render CLI; resize pixel centers.
    intrinsics = torch.eye(4, device=device).unsqueeze(0)
    intrinsics[0, 0, 0] = focal * rw / w
    intrinsics[0, 1, 1] = focal * rh / h
    intrinsics[0, 0, 2] = (rw - 1) / 2
    intrinsics[0, 1, 2] = (rh - 1) / 2
    extrinsics = torch.eye(4, device=device).unsqueeze(0)
    renderer = GSplatRenderer(color_space="linearRGB", background_color="black")

    def render(colors):
        return renderer(base._replace(colors=colors), extrinsics, intrinsics, rw, rh).color

    with torch.no_grad():
        original = render(base.colors)
    source_tensor = image_tensor(source, device, (rh, rw))
    save_image(source_tensor, args.output / "source.png")
    save_image(original, args.output / "baseline.png")
    save_image((original - source_tensor).abs(), args.output / "baseline_error.png")
    summary = {"baseline_vs_source": metrics(original, source_tensor), "targets": []}

    for index, path in enumerate(targets):
        folder = args.output / f"{index:02d}_{path.stem}"
        folder.mkdir()
        with Image.open(path) as image:
            target = image_tensor(np.asarray(ImageOps.exif_transpose(image).convert("RGB")),
                                  device, (rh, rw))
        save_image(target, folder / "target.png")
        # Independent restart for every target. Optimize RGB directly with projection
        # to [0,1], avoiding near-black sigmoid saturation during this capacity test.
        colors = torch.nn.Parameter(base.colors.clone().clamp(0, 1))
        optimizer = torch.optim.Adam([colors], lr=args.lr)
        before = metrics(original, target)
        with (folder / "loss.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "l1_before_update"])
            for step in range(1, args.steps + 1):
                optimizer.zero_grad(set_to_none=True)
                pred = render(colors)
                loss = F.l1_loss(pred, target)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at {path}, step {step}")
                loss.backward()
                if colors.grad is None or not torch.isfinite(colors.grad).all():
                    raise RuntimeError("Missing/non-finite color gradient")
                if step == 1 and colors.grad.abs().max().item() == 0:
                    raise RuntimeError("Zero color gradient; check camera/visibility")
                optimizer.step()
                with torch.no_grad():
                    colors.clamp_(0, 1)
                writer.writerow([step, loss.item()])
                if step == 1 or step % args.save_every == 0 or step == args.steps:
                    with torch.no_grad():
                        current = render(colors)
                    save_image(current, folder / f"step_{step:05d}.png")
                    print(f"{path.name} {step}/{args.steps}: L1={loss.item():.6f}", flush=True)
                    handle.flush()
        with torch.no_grad():
            final = render(colors)
        save_image(final, folder / "fitted.png")
        save_image((final - target).abs(), folder / "error.png")
        save_image(torch.cat([original, target, final, (final - target).abs()], dim=3),
                   folder / "comparison.png")
        torch.save(colors.detach().cpu(), folder / "colors_linear.pt")
        save_ply(base._replace(colors=colors.detach()), focal, (h, w), folder / "fitted.ply")
        summary["targets"].append({"target": str(path), "before": before,
                                    "after": metrics(final, target)})
        (args.output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        del optimizer, colors, pred, loss, current, final, target
    print(f"Done: {args.output}", flush=True)


if __name__ == "__main__":
    main()
