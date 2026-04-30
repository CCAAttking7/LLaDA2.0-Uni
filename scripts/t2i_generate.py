"""
LLaDA-2.0-Uni — Text-to-Image Generation

Usage:
    python t2i_generate.py --model_path /path/to/LLaDA-2.0-Uni --prompt "A cat on a table"
    python t2i_generate.py --model_path /path/to/LLaDA-2.0-Uni --prompts_file prompts.txt
"""

import os, sys, gc, argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decoder import decode_vq_tokens


def parse_args():
    p = argparse.ArgumentParser(description="LLaDA-2.0-Uni Text-to-Image Generation")
    p.add_argument("--model_path", type=str, required=True,
                   help="Root model dir containing LLM weights, image_tokenizer/, decoder/, vae/")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--prompts_file", type=str, default=None, help="One prompt per line")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--cfg_scale", type=float, default=4.0)
    p.add_argument("--image_h", type=int, default=512)
    p.add_argument("--image_w", type=int, default=512)
    p.add_argument("--decoder_steps", type=int, default=50)
    p.add_argument("--resolution_multiplier", type=int, default=2)
    p.add_argument("--output_dir", type=str, default="./t2i_output")
    p.add_argument("--output", type=str, default=None, help="Output path for single prompt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stability_probe", action="store_true",
                   help="Enable stability-layering hook and auto-save plots during generation")
    p.add_argument("--stability_layer_index", type=int, default=-2)
    p.add_argument("--stability_window", type=int, default=4)
    p.add_argument("--stability_attn_threshold", type=float, default=0.90)
    p.add_argument("--stability_conf_threshold", type=float, default=0.70)
    p.add_argument("--stability_anchor_patience", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompts = []
    if args.prompt: prompts = [args.prompt]
    elif args.prompts_file:
        with open(args.prompts_file) as f: prompts = [l.strip() for l in f if l.strip()]
    else: raise ValueError("--prompt or --prompts_file required")
    os.makedirs(args.output_dir, exist_ok=True)

    # Phase 1: generate VQ tokens
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map={"": device}, trust_remote_code=True)
    model = model.to(torch.bfloat16).eval()
    model.tokenizer = tokenizer

    results = []
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {prompt[:80]}")
        safe = prompt[:40].replace(" ", "_").replace("/", "")
        stability_plot_path = None
        stability_json_path = None
        if args.stability_probe:
            stability_plot_path = os.path.join(
                args.output_dir, f"{i:04d}_{safe}_stability.png"
            )
            stability_json_path = os.path.join(
                args.output_dir, f"{i:04d}_{safe}_stability.json"
            )
        res = model.generate_image(prompt, image_h=args.image_h, image_w=args.image_w,
                                   steps=args.steps, cfg_scale=args.cfg_scale,
                                   stability_probe=args.stability_probe,
                                   stability_plot_path=stability_plot_path,
                                   stability_json_path=stability_json_path,
                                   stability_layer_index=args.stability_layer_index,
                                   stability_window=args.stability_window,
                                   stability_attn_threshold=args.stability_attn_threshold,
                                   stability_conf_threshold=args.stability_conf_threshold,
                                   stability_anchor_patience=args.stability_anchor_patience)
        if args.stability_probe:
            print(f"  stability plot → {stability_plot_path}")
            print(f"  stability json → {stability_json_path}")
        results.append({"prompt": prompt, **res})

    del model; gc.collect(); torch.cuda.empty_cache()
    print("Model unloaded.\n")

    # Phase 2: decode to images
    for i, res in enumerate(results):
        if args.output and len(prompts) == 1:
            out = args.output
        else:
            safe = res["prompt"][:40].replace(" ", "_").replace("/", "")
            out = os.path.join(args.output_dir, f"{i:04d}_{safe}.png")
        print(f"[{i+1}/{len(results)}] Decoding → {out}")
        img = decode_vq_tokens(res["token_ids"], res["h"], res["w"], args.model_path, device,
                               resolution_multiplier=args.resolution_multiplier, num_steps=args.decoder_steps)
        img.save(out)

    print(f"\n🏁 Done! {len(results)} images generated.")


if __name__ == "__main__":
    main()
