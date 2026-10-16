"""Prepare / train / evaluate / infer SHARP-feature and U-Net relighters."""
import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from general_models import SceneRelighter, flat_to_map
from relight_mlp import normalize_positions
from train_relight import (ANGLES, check_target, choose, cuda_device, discover_sources,
                           file_digest, load_json, make_splits, read_target, save_image,
                           scene_renderer, target_path, write_json)


def load_sharp(path, device):
    from sharp.models import PredictorParams, create_predictor
    from sharp.cli.predict import DEFAULT_MODEL_URL
    state = (torch.load(path, map_location="cpu", weights_only=True) if path else
             torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, map_location="cpu", progress=True))
    model = create_predictor(PredictorParams())
    model.load_state_dict(state)
    return model.eval().requires_grad_(False).to(device)


@torch.no_grad()
def extract(model, source_path, device, feature_side):
    from sharp.cli.predict import predict_image
    from sharp.utils import io
    captured = {}

    def init_hook(module, inputs, output):
        # Actual [B,3,L,H,W] initialization, not an assumed sqrt(N).
        captured["grid"] = list(output.gaussian_base_values.colors.shape[2:])

    def feature_hook(module, inputs, output):
        value = torch.cat([output.texture_features, output.geometry_features], dim=1)
        captured["feature_shape_original"] = list(value.shape)
        if max(value.shape[-2:]) > feature_side:
            ratio = feature_side / max(value.shape[-2:])
            value = F.interpolate(value, size=(round(value.shape[-2] * ratio),
                                              round(value.shape[-1] * ratio)),
                                  mode="bilinear", align_corners=False)
        captured["features"] = value.cpu().half()

    h1 = model.init_model.register_forward_hook(init_hook)
    h2 = model.feature_model.register_forward_hook(feature_hook)
    try:
        source, _, focal = io.load_rgb(source_path)
        gs = predict_image(model, source, focal, device)
    finally:
        h1.remove()
        h2.remove()
    if math.prod(captured["grid"]) != gs.colors.shape[1]:
        raise ValueError("SHARP grid/Gaussian ordering mismatch")
    return dict(version=1, general_version=1, source=str(source_path.resolve()),
                source_sha256=file_digest(source_path), hw=list(source.shape[:2]),
                focal=float(focal), color_space="linearRGB", source_rgb=torch.from_numpy(source.copy()),
                gaussians={k: v.detach().cpu() for k, v in gs._asdict().items()}, **captured)


def prepare(args):
    from sharp.cli.predict import DEFAULT_MODEL_URL
    device = cuda_device()
    sources = discover_sources(args.source_root)
    scenes = choose(sources, args.scenes)
    identity = file_digest(args.sharp_checkpoint) if args.sharp_checkpoint else DEFAULT_MODEL_URL
    manifest = dict(general_version=1, feature_side=args.feature_side, checkpoint=identity,
                    source_root=str(args.source_root.resolve()), scenes=sorted(scenes),
                    splits=make_splits(scenes, args.seed))
    args.cache.mkdir(parents=True, exist_ok=True)
    manifest_path = args.cache / "manifest.json"
    if manifest_path.exists() and load_json(manifest_path) != manifest:
        raise ValueError("Use a new cache for different settings; old MLP caches lack features")
    write_json(manifest_path, manifest)
    pending = []
    for scene in scenes:
        file = args.cache / f"{scene}.pt"
        if file.exists():
            data = torch.load(file, map_location="cpu", weights_only=True)
            if data.get("general_version") != 1 or data["source_sha256"] != file_digest(sources[scene]):
                raise ValueError(f"Invalid/stale cache for {scene}")
            del data
        else:
            pending.append(scene)
    if not pending:
        print("All scenes already cached", flush=True)
        return
    model = load_sharp(args.sharp_checkpoint, device)
    for index, scene in enumerate(pending):
        print(f"Extract {index+1}/{len(pending)} {scene}", flush=True)
        data = extract(model, sources[scene], device, args.feature_side)
        temp = args.cache / f"{scene}.tmp"
        torch.save(data, temp)
        temp.replace(args.cache / f"{scene}.pt")
        del data


def inputs_from_data(data, gs, model_side, device, use_features):
    if data.get("general_version") != 1:
        raise ValueError("Run general_relight.py prepare; previous caches lack feature maps")
    grid = data["grid"]
    def resize(value):
        return F.interpolate(value, size=(model_side, model_side), mode="bilinear", align_corners=False)
    rgb = resize(flat_to_map(gs.colors, grid))
    xyz = resize(flat_to_map(normalize_positions(gs.mean_vectors), grid))
    alpha = resize(flat_to_map(gs.opacities.unsqueeze(-1), grid))
    image = data["source_rgb"].to(device=device, dtype=torch.float32).permute(2, 0, 1)[None] / 255
    # SHARP uses a square internal image; both features and grids share this domain.
    image = F.interpolate(image, size=(1536, 1536), mode="bilinear", align_corners=True)
    attributes = torch.cat([rgb, xyz, alpha, resize(image)], dim=1)
    features = data["features"].to(device=device, dtype=torch.float32) if use_features else None
    return attributes, features, grid


def load_scene(args, scene, model, device):
    gs, _, render, size, data = scene_renderer(args.cache, scene, args.max_side, device)
    attributes, features, grid = inputs_from_data(data, gs, args.model_side, device,
                                                model.architecture == "sharp")
    return gs, render, size, data, attributes, features, grid


def preflight(args, scenes):
    expected = None
    for scene in scenes:
        data = torch.load(args.cache / f"{scene}.pt", map_location="cpu", weights_only=True)
        if data.get("general_version") != 1:
            raise ValueError("Need general_relight.py prepare cache")
        signature = (data["grid"][0], data["features"].shape[1])
        if expected is not None and signature != expected:
            raise ValueError("Mixed SHARP grid layers or feature channels")
        expected = signature
        for angle in args.angles:
            check_target(target_path(args.target_root, scene, angle), data["hw"])
    return expected


@torch.no_grad()
def evaluate(model, args, scenes, output, device):
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    rows = []
    for scene in scenes:
        gs, render, size, data, attrs, feats, grid = load_scene(args, scene, model, device)
        baseline = render(gs.colors)
        for angle in args.angles:
            target = read_target(target_path(args.target_root, scene, angle), size, device)
            pred = render(model(attrs, feats, angle, grid))
            rows.append(dict(scene=scene, angle=angle, l1=F.l1_loss(pred,target).item(),
                             baseline_l1=F.l1_loss(baseline,target).item(),
                             psnr_db=-10*math.log10(max(F.mse_loss(pred,target).item(),1e-12))))
            save_image(torch.cat([baseline,target,pred,(pred-target).abs()],dim=3),
                       output / f"{scene}_{angle:03d}.png")
        print(f"Evaluated {scene}", flush=True)
    result = dict(pairs=rows, mean_l1=sum(r["l1"] for r in rows)/len(rows))
    write_json(output / "metrics.json", result)
    return result["mean_l1"]


def restore(path, device):
    saved = torch.load(path,map_location="cpu",weights_only=True)
    if saved.get("kind") != "general_relight_v1":
        raise ValueError("Not a general relighter checkpoint")
    model = SceneRelighter(**saved["model_config"]).to(device)
    model.load_state_dict(saved["model"])
    return model, saved


def train(args):
    device = cuda_device()
    manifest = load_json(args.cache / "manifest.json")
    scenes = choose(manifest["scenes"],args.scenes) if args.scenes else manifest["splits"]["train"]
    val = [] if args.scenes else manifest["splits"]["val"]
    if not scenes:
        raise ValueError("No training scenes")
    layers, feature_channels = preflight(args, scenes + val)
    signature = dict(manifest=manifest, scenes=scenes, angles=args.angles,
                     model_side=args.model_side, max_side=args.max_side, seed=args.seed,
                     architecture=args.architecture, target_root=str(args.target_root.resolve()))
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError("Use a new output directory or --resume")
    args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed)
    start,best=0,float("inf")
    if args.resume:
        model,saved=restore(args.resume,device)
        if saved["signature"] != signature:
            raise ValueError("Resume settings/cache differ")
        start,best=saved["epoch"],saved["best_val"]
    else:
        model=SceneRelighter(args.architecture,layers,feature_channels,args.width).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    if args.resume:
        optimizer.load_state_dict(saved["optimizer"])
        del saved
    if args.epochs <= start:
        raise ValueError("epochs must exceed saved epoch")
    write_json(args.output/"config.json",dict(signature=signature,model=model.config,lr=args.lr))
    print(f"{args.architecture}: {len(scenes)} train scenes, {len(val)} val scenes; "
          f"{sum(p.numel() for p in model.parameters()):,} shared parameters",flush=True)
    log=args.output/"loss.csv"
    header=not log.exists()
    with log.open("a",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle)
        if header: writer.writerow(["epoch","scene","angle","l1"])
        for epoch in range(start+1,args.epochs+1):
            model.train()
            rng=random.Random(args.seed+epoch)
            order=scenes.copy(); rng.shuffle(order)
            count,total=0,0.
            for scene in order:
                gs,render,size,data,attrs,feats,grid=load_scene(args,scene,model,device)
                angles=args.angles.copy(); rng.shuffle(angles)
                for angle in angles:
                    target=read_target(target_path(args.target_root,scene,angle),size,device)
                    optimizer.zero_grad(set_to_none=True)
                    pred=render(model(attrs,feats,angle,grid))
                    loss=F.l1_loss(pred,target)
                    if not torch.isfinite(loss): raise RuntimeError("Non-finite loss")
                    loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    if epoch==start+1 and count==0 and norm.item()==0:
                        raise RuntimeError("Zero relighter gradient")
                    optimizer.step()
                    writer.writerow([epoch,scene,angle,loss.item()])
                    total+=loss.item(); count+=1
                    if count==1 or count%args.log_every==0:
                        print(f"epoch={epoch} pair={count} {scene} angle={angle} L1={loss.item():.6f}",flush=True)
                    del pred,loss,target
                handle.flush()
            print(f"Epoch {epoch} train L1={total/count:.6f}",flush=True)
            improved=False
            if epoch%args.eval_every==0 or epoch==args.epochs:
                score=evaluate(model,args,val or scenes[:1],
                               args.output/f"{'val' if val else 'train_preview'}_{epoch:04d}",device)
                if val and score<best: best,improved=score,True
            payload=dict(kind="general_relight_v1",model_config=model.config,model=model.state_dict(),
                         optimizer=optimizer.state_dict(),epoch=epoch,best_val=best,signature=signature)
            temp=args.output/"last.tmp"; torch.save(payload,temp); temp.replace(args.output/"last.pt")
            if improved: torch.save(payload,args.output/"best.pt")
            del payload


def eval_command(args):
    device=cuda_device()
    model,saved=restore(args.model,device)
    args.model_side=saved["signature"]["model_side"]
    del saved
    manifest=load_json(args.cache/"manifest.json")
    scenes=choose(manifest["scenes"],args.scenes) if args.scenes else manifest["splits"][args.split]
    if not scenes: raise ValueError("Empty evaluation split")
    if args.output.exists() and any(args.output.iterdir()): raise ValueError("Use a new output directory")
    preflight(args,scenes)
    evaluate(model,args,scenes,args.output,device)


@torch.no_grad()
def infer(args):
    from sharp.cli.predict import DEFAULT_MODEL_URL
    from sharp.utils.gaussians import save_ply
    device=cuda_device()
    model,saved=restore(args.model,device)
    manifest=saved["signature"]["manifest"]
    args.model_side=saved["signature"]["model_side"]
    identity=file_digest(args.sharp_checkpoint) if args.sharp_checkpoint else DEFAULT_MODEL_URL
    if identity!=manifest["checkpoint"]:
        raise ValueError("Use the same SHARP checkpoint option as preparation")
    del saved
    if args.output.exists() and any(args.output.iterdir()): raise ValueError("Use a new output directory")
    args.output.mkdir(parents=True,exist_ok=True)
    sharp=load_sharp(args.sharp_checkpoint,device)
    data=extract(sharp,args.source,device,manifest["feature_side"])
    del sharp
    torch.cuda.empty_cache()
    # Retain a reusable scene cache; no fitting or target image is needed.
    cache=args.output/"scene_cache"; cache.mkdir()
    torch.save(data,cache/"input.pt")
    args.cache=cache
    model.eval()
    gs,render,size,data,attrs,feats,grid=load_scene(args,"input",model,device)
    if grid[0]!=model.config["layers"]: raise ValueError("SHARP layer count differs")
    colors=model(attrs,feats,args.angle,grid)
    save_image(render(gs.colors),args.output/"baseline.png")
    save_image(render(colors),args.output/"relit.png")
    save_ply(gs._replace(colors=colors),data["focal"],tuple(data["hw"]),args.output/"relit.ply")
    torch.save(colors.cpu(),args.output/"colors_linear.pt")
    write_json(args.output/"info.json",dict(source=str(args.source),angle=args.angle,model=str(args.model),
                                          architecture=model.architecture))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest="command",required=True)
    for name in ["prepare","train","evaluate","infer"]:
        cmd=subs.add_parser(name)
        if name!="infer": cmd.add_argument("--cache",type=Path,default=Path("relight_feature_cache"))
        if name in ("prepare","infer"):
            cmd.add_argument("--sharp-checkpoint",type=Path)
        if name=="prepare":
            cmd.add_argument("--source-root",type=Path,default=Path("flux_relight_source"))
            cmd.add_argument("--feature-side",type=int,default=384)
            cmd.add_argument("--scenes",nargs="+")
            cmd.add_argument("--seed",type=int,default=42)
        else:
            cmd.add_argument("--output",type=Path,required=True)
            cmd.add_argument("--max-side",type=int,default=512)
        if name in ("train","evaluate"):
            cmd.add_argument("--target-root",type=Path,default=Path("flux_relight_dataset"))
            cmd.add_argument("--scenes",nargs="+")
            cmd.add_argument("--angles",nargs="+",type=int,default=ANGLES)
        if name=="train":
            cmd.add_argument("--architecture",choices=["sharp","unet"],required=True)
            cmd.add_argument("--model-side",type=int,default=384)
            cmd.add_argument("--width",type=int,default=32)
            cmd.add_argument("--epochs",type=int,default=30)
            cmd.add_argument("--lr",type=float,default=0.0003)
            cmd.add_argument("--seed",type=int,default=42)
            cmd.add_argument("--eval-every",type=int,default=5)
            cmd.add_argument("--log-every",type=int,default=20)
            cmd.add_argument("--resume",type=Path)
        if name in ("evaluate","infer"): cmd.add_argument("--model",type=Path,required=True)
        if name=="evaluate": cmd.add_argument("--split",choices=["train","val","test"],default="test")
        if name=="infer":
            cmd.add_argument("--source",type=Path,required=True)
            cmd.add_argument("--angle",type=float,required=True)
    args=parser.parse_args()
    for key in ["max_side","feature_side","width","epochs","lr","eval_every","log_every"]:
        if hasattr(args,key) and getattr(args,key)<=0: parser.error(f"{key} must be positive")
    if hasattr(args,"width") and args.width%8: parser.error("width must be a multiple of 8")
    if hasattr(args,"model_side") and args.model_side<16: parser.error("model-side must be >=16")
    if hasattr(args,"angles"):
        if any(a not in ANGLES for a in args.angles): parser.error("Invalid dataset angle")
        args.angles=sorted(set(args.angles))
    if hasattr(args,"angle") and not 0<=args.angle<=180: parser.error("angle must be in [0,180]")
    {"prepare":prepare,"train":train,"evaluate":eval_command,"infer":infer}[args.command](args)


if __name__=="__main__": main()
