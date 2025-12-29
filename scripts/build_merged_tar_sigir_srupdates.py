import argparse
import os

import datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build and save the merged tar/sigir/sr_updates dataset in arrow format."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="data/merged_tar_sigir_srupdates",
        help="Directory to save the dataset (arrow format).",
    )
    parser.add_argument(
        "--config",
        default="merged_tar_sigir_srupdates_all_source",
        help="Dataset config name to build.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise ValueError(
            f"Output directory is not empty: {output_dir}. Choose a new path."
        )

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csmed_root = os.path.join(repo_root, "csmed")
    os.environ.setdefault("CSMED_ROOT", csmed_root)

    dataset_path = os.path.join(
        csmed_root,
        "datasets",
        "merged_tar_sigir_srupdates",
        "merged_tar_sigir_srupdates.py",
    )
    dataset = datasets.load_dataset(
        dataset_path, name=args.config, trust_remote_code=True
    )
    dataset.save_to_disk(output_dir)
    print(f"Saved dataset to {output_dir}")


if __name__ == "__main__":
    main()
