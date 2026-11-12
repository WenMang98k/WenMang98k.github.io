"""Original research reference by WenMang, not an executed GPU benchmark.
Dry-run needs only Python; actual generation needs compatible PyTorch, Diffusers,
Transformers, model access, suitable hardware and applicable model permissions.
The provided Lightning model card was not read; no Lightning recipe is invented.
"""
import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='Qwen-Image-2.1 Base/Turbo reference')
    p.add_argument('--variant', choices=['base', 'turbo'], default='turbo')
    p.add_argument('--prompt', default='一幅清新的城市公园插画，自然光，舒展的构图。')
    p.add_argument('--image', type=Path, nargs='*', default=[])
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--height', type=int, default=1024)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--offload', action='store_true')
    p.add_argument('--output', type=Path, default=Path('qwen21-output.png'))
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if any(v < 32 or v % 32 for v in [args.width, args.height]):
        p.error('width and height must be positive multiples of 32, at least 32')
    if len(args.image) > 10:
        p.error('This reference limits condition images to the documented maximum of 10')
    if args.output.suffix.lower() != '.png':
        p.error('Use PNG to retain any alpha channel produced by the model')
    return args


def main():
    args = parse_args()
    model_id = 'Qwen/Qwen-Image-2.1' + ('-Turbo' if args.variant == 'turbo' else '')
    plan = {
        'model_id': model_id,
        'variant': args.variant,
        'task': 'i2i' if args.image else 't2i',
        'reference_image_count': len(args.image),
        'width': args.width, 'height': args.height,
        'target_latent_tokens_estimate': (args.width // 16) * (args.height // 16),
        'sampling': 'checkpoint sample_sigmas (official 8-step Turbo)' if args.variant == 'turbo' else '40 steps, true_cfg_scale=1.0',
        'placement': 'model CPU offload' if args.offload else 'CUDA full placement',
        'lightning': 'not loaded: model-card recipe not verified',
        'execution_requested': not args.dry_run,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    for file in args.image:
        if not file.is_file():
            raise FileNotFoundError(file)
    import torch
    from PIL import Image
    from diffusers import QwenImage21Pipeline
    if not torch.cuda.is_available():
        raise RuntimeError('This CUDA reference requires a compatible GPU; no silent CPU fallback')
    pipe = QwenImage21Pipeline.from_pretrained(model_id, dtype=torch.bfloat16)
    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to('cuda')
    call = dict(prompt=args.prompt, width=args.width, height=args.height,
                generator=torch.Generator('cuda').manual_seed(args.seed))
    if args.variant == 'base':
        call.update(num_inference_steps=40, true_cfg_scale=1.0)
    # Turbo deliberately does not override the checkpoint's default sigma grid.
    if args.image:
        inputs = []
        for file in args.image:
            with Image.open(file) as source:
                inputs.append(source.convert('RGBA'))
        call['image'] = inputs
    image = pipe(**call).images[0]
    print('Actual denoising timesteps:', pipe.num_timesteps)
    print('Checkpoint sample_sigmas:', getattr(pipe.config, 'sample_sigmas', None))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print('Saved:', args.output, 'mode:', image.mode)
    if 'A' in image.getbands():
        print('Alpha range:', image.getchannel('A').getextrema(), '(constant 255 means fully opaque)')


if __name__ == '__main__':
    main()
