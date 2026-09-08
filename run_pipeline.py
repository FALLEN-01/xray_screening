"""
run_pipeline.py — One-command end-to-end pipeline runner.
Runs: synthetic data gen → train → evaluate → gradcam → demo app

Usage:
    python run_pipeline.py                   # full pipeline
    python run_pipeline.py --skip-train      # evaluate + gradcam only
    python run_pipeline.py --demo-only       # just launch the web app
"""
import os
import sys
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd: str, desc: str):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(
        cmd, shell=True, cwd=ROOT
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

    cfg_flag = f"--config {args.config}"

    if args.demo_only:
        run(f"python app/demo.py {cfg_flag}", "Launching Gradio Demo App")
        return

    if not args.skip_train:
        run(f"python src/train.py {cfg_flag} --demo",
            "Step 1/3 — Generate Synthetic Data + Train Model")

    run(f"python src/evaluate.py {cfg_flag}",
        "Step 2/3 — Evaluate on Test Set")

    run(f"python src/gradcam.py {cfg_flag}",
        "Step 3/3 — Generate Grad-CAM Visualizations")

    print("\n" + "="*60)
    print("  Pipeline complete!")
    print("  Outputs:")
    print("    outputs/checkpoints/best_model.pth")
    print("    outputs/logs/training_log.csv")
    print("    outputs/visualizations/")
    print("="*60)
    print("\nLaunch the web demo with:")
    print("    python app/demo.py\n")


if __name__ == "__main__":
    main()
