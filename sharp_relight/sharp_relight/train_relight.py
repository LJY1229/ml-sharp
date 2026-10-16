"""SHARP relighting baseline: prepare / train / evaluate / infer (one CUDA GPU)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from relight_mlp import RelightMLP, normalize_positions, predict_colors

ANGLES = [0, 30, 60, 90, 120, 150, 180]
VERSION = 1


def file_digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def target_path(root, scene, angle):
    return root / f"angle_{angle:03d}" / f"{scene}_angle_{angle:03d}.png"


def discover_sources(root):
    found = {}
    for category in ("indoor", "outdoor"):
        for path in sorted((root / category).glob(f"{category}_*/source.jpg")):
            found[path.parent.name] = path
    if not found:
        raise ValueError(f"No indoor/outdoor scene folders under {root}")
    return found


def make_splits(scenes, seed=42):
    rng = random.Random(seed)
    result = {"train": [], "val": [], "test": []}
    for category in ("indoor", "outdoor"):
        group = sorted(s for s in scenes if s.startswith(category + "_"))
        rng.shuffle(group)
        count = len(group) // 10
        result["val"].extend(group[:count])
        result["test"].extend(group[count:2 * count])
        result["train"].extend(group[2 * count:])
    return result


def choose(scenes, requested):
    if not requested:
        return list(scenes)
    missing = set(requested) - set(scenes)
    if missing:
        raise ValueError(f"Unknown scenes: {sorted(missing)}")
    return list(dict.fromkeys(requested))


def check_target(path, source_hw):
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        width, height = ImageOps.exif_transpose(image).size
    h, w = source_hw
    if abs((width / height) / (w / h) - 1) > 0.02:
        raise ValueError(f"Aspect ratio mismatch: {path}: {width}x{height}, source {w}x{h}. "
                         "Align/crop the pair before training; automatic stretching is disabled.")


def read_target(path, size, device):
    with Image.open(path) as image:
        array = np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
    tensor = torch.from_numpy(array).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1)[None] / 255
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False, antialias=True)


def save_image(tensor, path):
    array = tensor.detach()[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    Image.fromarray(np.round(array * 255).astype(np.uint8)).save(path)


def cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("Run this command in the installed SHARP CUDA environment on your H100.")
    return torch.device("cuda")


def prepare(args):
    from sharp.cli.predict import DEFAULT_MODEL_URL, predict_image
    from sharp.models import PredictorParams, create_predictor
    from sharp.utils import io

    device = cuda_device()
    sources = discover_sources(args.source_root)
    selected = choose(sources, args.scenes)
    args.cache.mkdir(parents=True, exist_ok=True)
    checkpoint_id = (file_digest(args.sharp_checkpoint)
                     if args.sharp_checkpoint else DEFAULT_MODEL_URL)
    manifest = {"version": VERSION, "source_root": str(args.source_root.resolve()),
                "checkpoint_id": checkpoint_id, "scenes": sorted(selected),
                "splits": make_splits(selected, args.seed)}
    manifest_path = args.cache / "manifest.json"
    if manifest_path.exists() and load_json(manifest_path) != manifest:
        raise ValueError("Cache configuration differs. Use a new cache directory.")
    # Write before inference to support restarting an interrupted preparation.
    write_json(manifest_path, manifest)
    pending = []
    for scene in selected:
        digest = file_digest(sources[scene])
        file = args.cache / f"{scene}.pt"
        if file.exists():
            saved = torch.load(file, map_location="cpu", weights_only=True)
            if saved["source_sha256"] != digest:
                raise ValueError(f"Source changed: {scene}. Use a new cache directory.")
            print(f"Reuse {scene}", flush=True)
        else:
            pending.append((scene, digest))
    if not pending:
        return
    state = (torch.load(args.sharp_checkpoint, map_location="cpu", weights_only=True)
             if args.sharp_checkpoint else torch.hub.load_state_dict_from_url(
                 DEFAULT_MODEL_URL, map_location="cpu", progress=True))
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state)
    predictor.eval().requires_grad_(False).to(device)
    del state
    for index, (scene, digest) in enumerate(pending):
        source, _, focal = io.load_rgb(sources[scene])
        print(f"Predict {index + 1}/{len(pending)} {scene}", flush=True)
        with torch.no_grad():
            gs = predict_image(predictor, source, focal, device)
        data = {"version": VERSION, "source_sha256": digest,
                "source": str(sources[scene].resolve()), "hw": list(source.shape[:2]),
                "focal": float(focal), "color_space": "linearRGB",
                "gaussians": {k: v.detach().cpu() for k, v in gs._asdict().items()}}
        temporary = args.cache / f"{scene}.tmp"
        torch.save(data, temporary)
        temporary.replace(args.cache / f"{scene}.pt")
        del gs, data


def scene_renderer(cache, scene, max_side, device):
    from sharp.utils.gaussians import Gaussians3D
    from sharp.utils.gsplat import GSplatRenderer

    data = torch.load(cache / f"{scene}.pt", map_location="cpu", weights_only=True)
    if data["version"] != VERSION or data["color_space"] != "linearRGB":
        raise ValueError("Unsupported scene cache")
    gs = Gaussians3D(**{key: value.to(device).detach()
                       for key, value in data["gaussians"].items()})
    h, w = data["hw"]
    ratio = min(1., max_side / max(h, w))
    rh, rw = max(1, round(h * ratio)), max(1, round(w * ratio))
    k = torch.eye(4, device=device)[None]
    k[0, 0, 0], k[0, 1, 1] = data["focal"] * rw / w, data["focal"] * rh / h
    k[0, 0, 2], k[0, 1, 2] = (rw - 1) / 2, (rh - 1) / 2
    pose = torch.eye(4, device=device)[None]
    renderer = GSplatRenderer(color_space="linearRGB", background_color="black")

    def render(colors):
        return renderer(gs._replace(colors=colors), pose, k, rw, rh).color

    xyz = normalize_positions(gs.mean_vectors)
    return gs, xyz, render, (rh, rw), data


def preflight(cache, root, scenes, angles):
    for scene in scenes:
        data = torch.load(cache / f"{scene}.pt", map_location="cpu", weights_only=True)
        for angle in angles:
            check_target(target_path(root, scene, angle), data["hw"])


@torch.no_grad()
def evaluate(model, args, scenes, angles, output, device):
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    rows = []
    for scene in scenes:
        gs, xyz, render, size, _ = scene_renderer(args.cache, scene, args.max_side, device)
        baseline = render(gs.colors)
        for angle in angles:
            target = read_target(target_path(args.target_root, scene, angle), size, device)
            colors = predict_colors(model, xyz, gs.colors, angle, args.chunk_size)
            pred = render(colors)
            mse = F.mse_loss(pred, target).item()
            rows.append({"scene": scene, "angle": angle,
                         "l1": F.l1_loss(pred, target).item(),
                         "baseline_l1": F.l1_loss(baseline, target).item(),
                         "psnr_db": -10 * math.log10(max(mse, 1e-12))})
            save_image(torch.cat([baseline, target, pred, (pred - target).abs()], dim=3),
                       output / f"{scene}_{angle:03d}.png")
        print(f"Evaluated {scene}", flush=True)
    result = {"pairs": rows, "mean_l1": sum(r["l1"] for r in rows) / len(rows),
              "mean_baseline_l1": sum(r["baseline_l1"] for r in rows) / len(rows)}
    write_json(output / "metrics.json", result)
    return result["mean_l1"]


def load_model(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = RelightMLP(**saved["model_config"]).to(device)
    model.load_state_dict(saved["model"])
    return model, saved


def train(args):
    device = cuda_device()
    torch.manual_seed(args.seed)
    manifest = load_json(args.cache / "manifest.json")
    # Explicit --scenes intentionally enables a single-scene overfit experiment.
    scenes = (choose(manifest["scenes"], args.scenes) if args.scenes
              else manifest["splits"]["train"])
    val_scenes = [] if args.scenes else manifest["splits"]["val"]
    if not scenes:
        raise ValueError("No training scenes")
    preflight(args.cache, args.target_root, scenes, args.angles)
    if val_scenes:
        preflight(args.cache, args.target_root, val_scenes, args.angles)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError("Use a new output directory or --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    start, best = 0, float("inf")
    if args.resume:
        model, saved = load_model(args.resume, device)
        if saved["train_scenes"] != scenes or saved["angles"] != args.angles:
            raise ValueError("Resume requires identical training scenes and angles")
        if saved["cache_manifest"] != manifest:
            raise ValueError("Resume cache differs")
        if saved["max_side"] != args.max_side or saved["seed"] != args.seed:
            raise ValueError("Resume requires identical max-side and seed")
        start, best = saved["epoch"], saved["best_val"]
    else:
        model = RelightMLP(args.hidden, args.layers, args.frequencies).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.resume:
        optimizer.load_state_dict(saved["optimizer"])
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key != "func"}
    config.update(train_scenes=scenes, val_scenes=val_scenes, model_config=model.config)
    write_json(args.output / "config.json", config)
    print(f"Train {len(scenes)} scenes x {len(args.angles)} angles; "
          f"validation {len(val_scenes)} scenes; parameters "
          f"{sum(p.numel() for p in model.parameters()):,}", flush=True)
    log_path = args.output / "loss.csv"
    needs_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if needs_header:
            writer.writerow(["epoch", "scene", "angle", "l1"])
        for epoch in range(start + 1, args.epochs + 1):
            model.train()
            rng = random.Random(args.seed + epoch)
            order = scenes.copy()
            rng.shuffle(order)
            total, count = 0., 0
            for scene in order:
                gs, xyz, render, size, _ = scene_renderer(args.cache, scene, args.max_side, device)
                angles = args.angles.copy()
                rng.shuffle(angles)
                for angle in angles:
                    target = read_target(target_path(args.target_root, scene, angle), size, device)
                    optimizer.zero_grad(set_to_none=True)
                    colors = predict_colors(model, xyz, gs.colors, angle, args.chunk_size)
                    pred = render(colors)
                    loss = F.l1_loss(pred, target)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss: {scene}, angle {angle}")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    if epoch == start + 1 and count == 0 and norm.item() == 0:
                        raise RuntimeError("Zero network gradient: check visibility and renderer")
                    optimizer.step()
                    total += loss.item()
                    count += 1
                    writer.writerow([epoch, scene, angle, loss.item()])
                    if count == 1 or count % args.log_every == 0:
                        print(f"epoch={epoch} pair={count}/{len(scenes)*len(angles)} "
                              f"{scene} angle={angle} L1={loss.item():.6f}", flush=True)
                    del colors, pred, loss, target
                handle.flush()
            print(f"Epoch {epoch}: train L1={total/count:.6f}", flush=True)
            improved = False
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                # In overfit mode previews are TRAINING examples, not validation.
                preview_scenes = val_scenes or scenes[:1]
                score = evaluate(model, args, preview_scenes, args.angles,
                                 args.output / f"{'val' if val_scenes else 'train_preview'}_{epoch:04d}", device)
                if val_scenes and score < best:
                    best, improved = score, True
            saved = {"version": VERSION, "epoch": epoch, "model_config": model.config,
                     "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "best_val": best, "train_scenes": scenes, "angles": args.angles,
                     "cache_manifest": manifest, "max_side": args.max_side, "seed": args.seed}
            temp = args.output / "last.tmp"
            torch.save(saved, temp)
            temp.replace(args.output / "last.pt")
            if improved:
                torch.save(saved, args.output / "best.pt")


def eval_command(args):
    device = cuda_device()
    model, _ = load_model(args.model, device)
    manifest = load_json(args.cache / "manifest.json")
    scenes = (choose(manifest["scenes"], args.scenes) if args.scenes
              else manifest["splits"][args.split])
    if not scenes:
        raise ValueError("Empty split; supply --scenes for a small-cache experiment")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new evaluation output directory")
    preflight(args.cache, args.target_root, scenes, args.angles)
    evaluate(model, args, scenes, args.angles, args.output, device)


@torch.no_grad()
def infer(args):
    from sharp.utils.gaussians import save_ply

    device = cuda_device()
    model, _ = load_model(args.model, device)
    model.eval()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new inference output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    gs, xyz, render, _, data = scene_renderer(args.cache, args.scene, args.max_side, device)
    colors = predict_colors(model, xyz, gs.colors, args.angle, args.chunk_size)
    save_image(render(gs.colors), args.output / "baseline.png")
    save_image(render(colors), args.output / "relit.png")
    torch.save(colors.cpu(), args.output / "colors_linear.pt")
    save_ply(gs._replace(colors=colors), data["focal"], tuple(data["hw"]),
             args.output / "relit.ply")
    write_json(args.output / "info.json", {"scene": args.scene, "angle": args.angle,
                                          "model": str(args.model)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source-root", type=Path, default=Path("flux_relight_source"))
    prep.add_argument("--cache", type=Path, default=Path("relight_cache"))
    prep.add_argument("--sharp-checkpoint", type=Path)
    prep.add_argument("--scenes", nargs="+")
    prep.add_argument("--seed", type=int, default=42)
    prep.set_defaults(func=prepare)
    for name, func in [("train", train), ("evaluate", eval_command), ("infer", infer)]:
        cmd = commands.add_parser(name)
        cmd.set_defaults(func=func)
        cmd.add_argument("--cache", type=Path, default=Path("relight_cache"))
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--max-side", type=int, default=512)
        cmd.add_argument("--chunk-size", type=int, default=65536)
        if name != "infer":
            cmd.add_argument("--target-root", type=Path, default=Path("flux_relight_dataset"))
            cmd.add_argument("--scenes", nargs="+")
            cmd.add_argument("--angles", nargs="+", type=int, default=ANGLES)
        if name == "train":
            cmd.add_argument("--epochs", type=int, default=30)
            cmd.add_argument("--lr", type=float, default=1e-3)
            cmd.add_argument("--hidden", type=int, default=64)
            cmd.add_argument("--layers", type=int, default=3)
            cmd.add_argument("--frequencies", type=int, default=4)
            cmd.add_argument("--seed", type=int, default=42)
            cmd.add_argument("--eval-every", type=int, default=5)
            cmd.add_argument("--log-every", type=int, default=20)
            cmd.add_argument("--resume", type=Path)
        else:
            cmd.add_argument("--model", type=Path, required=True)
        if name == "evaluate":
            cmd.add_argument("--split", choices=["train", "val", "test"], default="val")
        if name == "infer":
            cmd.add_argument("--scene", required=True)
            cmd.add_argument("--angle", type=float, required=True)
    args = parser.parse_args()
    for key in ["max_side", "chunk_size", "epochs", "lr", "hidden", "layers", "eval_every", "log_every"]:
        if hasattr(args, key) and getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    if hasattr(args, "frequencies") and args.frequencies < 0:
        parser.error("frequencies cannot be negative")
    if hasattr(args, "angles"):
        if any(angle not in ANGLES for angle in args.angles):
            parser.error(f"Dataset angles must be selected from {ANGLES}")
        args.angles = sorted(set(args.angles))
    if hasattr(args, "angle") and not 0 <= args.angle <= 180:
        parser.error("Inference angle must be in [0, 180] for this experiment")
    args.func(args)


if __name__ == "__main__":
    main()
