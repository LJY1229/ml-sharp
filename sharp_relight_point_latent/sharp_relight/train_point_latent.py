"""Train a single-scene latent-feature relighter directly from FLUX images."""
import argparse
import csv
import math
import random

from pathlib import Path
import torch
import torch.nn.functional as F

from point_latent import PointLatentRelighter
from train_relight import (ANGLES, check_target, cuda_device, file_digest, read_target,
                           save_image, scene_renderer, target_path, write_json)


def sample_angle(step, angles, seed):
    order = list(angles)
    random.Random(seed + (step - 1) // len(order)).shuffle(order)
    return order[(step - 1) % len(order)]


@torch.no_grad()
def preview(model, render, baseline, targets, folder, chunk_size):
    folder.mkdir(parents=True, exist_ok=True)
    model.eval()
    results = []
    for angle, target in targets.items():
        pred = render(model(angle, chunk_size))
        mse = F.mse_loss(pred, target).item()
        results.append(dict(angle=angle, l1=F.l1_loss(pred, target).item(),
                            baseline_l1=F.l1_loss(baseline, target).item(),
                            psnr_db=-10 * math.log10(max(mse, 1e-12))))
        save_image(torch.cat([baseline, target, pred, (pred-target).abs()], dim=3),
                   folder / f"angle_{angle:03d}.png")
    write_json(folder / "metrics.json", results)
    model.train()


def load_checkpoint(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("kind") != "point_latent_v1":
        raise ValueError("Expected a point-latent checkpoint, not the earlier MLP checkpoint")
    model = PointLatentRelighter(**saved["model_config"]).to(device)
    model.load_state_dict(saved["model"])
    return model, saved


def train(args):
    device = cuda_device()
    cache_file = args.cache / f"{args.scene}.pt"
    digest = file_digest(cache_file)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError("Use a new output directory or --resume")
    gs, _, render, size, data = scene_renderer(args.cache, args.scene, args.max_side, device)
    targets = {}
    target_digests = {}
    for angle in args.angles:
        path = target_path(args.target_root, args.scene, angle)
        check_target(path, data["hw"])
        targets[angle] = read_target(path, size, device)
        target_digests[str(angle)] = file_digest(path)
    signature = dict(scene=args.scene, cache_sha256=digest, angles=args.angles,
                     targets=target_digests, max_side=args.max_side, seed=args.seed)
    torch.manual_seed(args.seed)
    start = 0
    if args.resume:
        model, saved = load_checkpoint(args.resume, device)
        if signature != saved["signature"]:
            raise ValueError("Resume requires the exact same scene cache, labels, angles, size and seed")
        start = saved["step"]
    else:
        model = PointLatentRelighter(gs.colors.shape[1], args.latent_dim,
                                    args.hidden, args.layers).to(device)
    optimizer = torch.optim.Adam([
        {"params": [model.features], "lr": args.feature_lr},
        {"params": model.decoder.parameters(), "lr": args.network_lr},
    ])
    if args.resume:
        optimizer.load_state_dict(saved["optimizer"])
        del saved
    if args.steps <= start:
        raise ValueError(f"--steps is total updates and must exceed resumed step {start}")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(model_config=model.config, signature=signature)
    write_json(args.output / "config.json", config)
    with torch.no_grad():
        baseline = render(gs.colors)
    save_image(baseline, args.output / "baseline.png")
    preview(model, render, baseline, targets, args.output / f"preview_{start:06d}", args.chunk_size)
    print(f"Train scene={args.scene}; {len(model.features):,} Gaussian features; "
          f"{len(args.angles)} angles; {args.steps} total updates. Only features and decoder train.", flush=True)
    log = args.output / "loss.csv"
    header = not log.exists()
    with log.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if header:
            writer.writerow(["step", "angle", "l1", "feature_grad_norm", "decoder_grad_norm"])
        model.train()
        for step in range(start + 1, args.steps + 1):
            angle = sample_angle(step, args.angles, args.seed)
            optimizer.zero_grad(set_to_none=True)
            colors = model(angle, args.chunk_size)
            pred = render(colors)
            loss = F.l1_loss(pred, targets[angle])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at {step}")
            loss.backward()
            feature_norm = torch.nn.utils.clip_grad_norm_([model.features], 10., error_if_nonfinite=True)
            network_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1., error_if_nonfinite=True)
            if step == start + 1 and (feature_norm.item() == 0 or network_norm.item() == 0):
                raise RuntimeError("Missing gradient in features or network: inspect renderer/visibility")
            optimizer.step()
            writer.writerow([step, angle, loss.item(), feature_norm.item(), network_norm.item()])
            if step == start + 1 or step % args.log_every == 0:
                print(f"step={step}/{args.steps} angle={angle} L1={loss.item():.6f}", flush=True)
                handle.flush()
            del colors, pred, loss
            if step % args.preview_every == 0 or step == args.steps:
                preview(model, render, baseline, targets,
                        args.output / f"preview_{step:06d}", args.chunk_size)
            if step % args.save_every == 0 or step == args.steps:
                payload = dict(kind="point_latent_v1", step=step, signature=signature,
                               model_config=model.config, model=model.state_dict(),
                               optimizer=optimizer.state_dict())
                temp = args.output / "last.tmp"
                torch.save(payload, temp)
                temp.replace(args.output / "last.pt")
                del payload


@torch.no_grad()
def infer(args):
    from sharp.utils.gaussians import save_ply

    device = cuda_device()
    model, saved = load_checkpoint(args.model, device)
    signature = saved["signature"]
    scene = signature["scene"]
    if file_digest(args.cache / f"{scene}.pt") != signature["cache_sha256"]:
        raise ValueError("This model requires its EXACT original Gaussian cache and ordering")
    del saved
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new inference output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    gs, _, render, size, data = scene_renderer(args.cache, scene, args.max_side, device)
    model.eval()
    colors = model(args.angle, args.chunk_size)
    pred, baseline = render(colors), render(gs.colors)
    save_image(pred, args.output / "relit.png")
    save_image(baseline, args.output / "baseline.png")
    torch.save(colors.cpu(), args.output / "colors_linear.pt")
    save_ply(gs._replace(colors=colors), data["focal"], tuple(data["hw"]), args.output / "relit.ply")
    info = dict(scene=scene, angle=args.angle, trained_angles=signature["angles"])
    if args.target_root is not None and args.angle in ANGLES:
        path = target_path(args.target_root, scene, int(args.angle))
        check_target(path, data["hw"])
        target = read_target(path, size, device)
        save_image(torch.cat([baseline, target, pred, (pred-target).abs()], dim=3),
                   args.output / "comparison.png")
        info.update(l1=F.l1_loss(pred, target).item(),
                    baseline_l1=F.l1_loss(baseline, target).item())
    write_json(args.output / "info.json", info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["train", "infer"]:
        cmd = commands.add_parser(name)
        cmd.add_argument("--cache", type=Path, default=Path("relight_cache_one"))
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--max-side", type=int, default=512)
        cmd.add_argument("--chunk-size", type=int, default=32768)
        cmd.add_argument("--target-root", type=Path,
                         default=Path("flux_relight_dataset") if name == "train" else None)
        if name == "train":
            cmd.add_argument("--scene", required=True)
            cmd.add_argument("--angles", type=int, nargs="+", default=ANGLES)
            cmd.add_argument("--steps", type=int, default=7000)
            cmd.add_argument("--latent-dim", type=int, default=32)
            cmd.add_argument("--hidden", type=int, default=128)
            cmd.add_argument("--layers", type=int, default=3)
            cmd.add_argument("--feature-lr", type=float, default=0.01)
            cmd.add_argument("--network-lr", type=float, default=0.001)
            cmd.add_argument("--log-every", type=int, default=50)
            cmd.add_argument("--preview-every", type=int, default=500)
            cmd.add_argument("--save-every", type=int, default=1000)
            cmd.add_argument("--seed", type=int, default=42)
            cmd.add_argument("--resume", type=Path)
        else:
            cmd.add_argument("--model", type=Path, required=True)
            cmd.add_argument("--angle", type=float, required=True)
    args = parser.parse_args()
    for key in ["max_side", "chunk_size", "steps", "latent_dim", "hidden", "layers",
                "feature_lr", "network_lr", "log_every", "preview_every", "save_every"]:
        if hasattr(args, key) and getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    if args.command == "train":
        if any(a not in ANGLES for a in args.angles):
            parser.error(f"Angles must be selected from {ANGLES}")
        args.angles = sorted(set(args.angles))
        train(args)
    else:
        if not 0 <= args.angle <= 180:
            parser.error("angle must be between 0 and 180")
        infer(args)


if __name__ == "__main__":
    main()
