#!/usr/bin/env python
"""Download a subset of a LIBERO task suite's demo files from Hugging Face.

Each task in a suite is a separate ~700-800MB hdf5 file. LIBERO's own
download script only offers whole-suite downloads (~7.5GB for libero_object),
but for the compute-minimized MVP we only want 3-4 tasks. This pulls exactly
the files we ask for.

Usage:
    uv run python scripts/download_libero_data.py --suite libero_object --num-tasks 3
    uv run python scripts/download_libero_data.py --suite libero_object --match salad_dressing butter cream_cheese
"""
import argparse
import os

from huggingface_hub import HfApi, hf_hub_download

HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        default="libero_object",
        choices=["libero_object", "libero_goal", "libero_spatial", "libero_100"],
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Download only the first N task files in the suite (alphabetical). "
        "Omit to download the whole suite.",
    )
    parser.add_argument(
        "--match",
        nargs="*",
        default=None,
        help="Only download files whose name contains one of these substrings, "
        "e.g. --match salad_dressing butter cream_cheese",
    )
    parser.add_argument(
        "--download-dir",
        default=os.environ.get(
            "LIBERO_DATASET_DIR",
            os.path.join(os.path.dirname(__file__), "..", "data", "libero_datasets"),
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    api = HfApi()
    all_files = api.list_repo_files(HF_REPO_ID, repo_type="dataset")
    suite_files = sorted(f for f in all_files if f.startswith(f"{args.suite}/"))

    if args.match:
        suite_files = [f for f in suite_files if any(m in f for m in args.match)]
    elif args.num_tasks is not None:
        suite_files = suite_files[: args.num_tasks]

    if not suite_files:
        raise SystemExit(f"No files matched for suite={args.suite}")

    print(f"Downloading {len(suite_files)}/{len(all_files)} files from {args.suite}:")
    for f in suite_files:
        print(f"  {f}")

    for f in suite_files:
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename=f,
            local_dir=args.download_dir,
        )

    print(f"Done. Files are under {os.path.abspath(args.download_dir)}")


if __name__ == "__main__":
    main()
