"""One-command YOLO training, evaluation, and visualization pipeline.

Usage:
    python run_pipeline.py                   # full pipeline
    python run_pipeline.py --skip-train      # evaluate + predictions only
    python run_pipeline.py --demo-only       # just launch the web app
"""
import os
import sys
import argparse
import subprocess

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd: list[str], desc: str):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(
        cmd, cwd=ROOT
    )
    if result.returncode != 0:
        print(f"[ERROR] Step failed: {desc}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--demo-only",  action="store_true")
    args = parser.parse_args()

    config_path = os.path.abspath(os.path.join(ROOT, args.config))
    if not os.path.isfile(config_path):
        parser.error(f"config file not found: {config_path}")

    python = sys.executable
    cfg_flag = ["--config", config_path]
    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    checkpoint = os.path.join(
        ROOT, config["outputs"]["best_model"]
    )

    if args.demo_only:
        if not os.path.isfile(checkpoint):
            parser.error(
                f"checkpoint not found: {checkpoint}. "
                "Train the model before using --demo-only."
            )
        run(
            [python, "app/demo.py", *cfg_flag],
            "Launching Gradio Demo App",
        )
        return

    if not args.skip_train:
        run(
            [python, "src/train.py", *cfg_flag],
            "Step 1/3 — Train YOLO Detector",
        )
    elif not os.path.isfile(checkpoint):
        parser.error(
            f"checkpoint not found: {checkpoint}. "
            "Run without --skip-train first."
        )

    run(
        [python, "src/evaluate.py", *cfg_flag],
        "Step 2/3 — Evaluate on Test Set",
    )

    run(
        [python, "src/gradcam.py", *cfg_flag],
        "Step 3/3 — Generate Detection Visualizations",
    )

    print("\n" + "="*60)
    print("  Pipeline complete!")
    print("  Outputs:")
    print("    outputs/checkpoints/detector/weights/best.pt")
    print("    outputs/visualizations/")
    print("="*60)
    print("\nLaunch the web demo with:")
    print("    python app/demo.py\n")


if __name__ == "__main__":
    main()
