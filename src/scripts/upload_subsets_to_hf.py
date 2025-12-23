import os
import json
from pathlib import Path
from datasets import load_from_disk, DatasetDict
from huggingface_hub import HfApi, create_repo

def upload_subsets(repo_id: str, subsets_dir: str = "subsets"):
    subsets_path = Path(subsets_dir)
    if not subsets_path.exists():
        print(f"Directory {subsets_dir} not found.")
        return

    if not os.getenv("HF_TOKEN"):
        print("Warning: HF_TOKEN environment variable not found. You might need to be logged in via `huggingface-cli login` or provide the token.")

    api = HfApi()
    
    # Try to create the repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        print(f"Repository {repo_id} ready.")
    except Exception as e:
        print(f"Note: Could not create/verify repo {repo_id}: {e}")

    readme_content = f"""# OpenThoughts Subsets (Forks)

This repository contains various subsets and forks of the [OpenThoughts](https://huggingface.co/open-thoughts) datasets. 
Each configuration here is a filtered version of a larger open-source dataset, optimized for specific token limits and sources.

## Dataset Structure
Each subset is available as a separate configuration. You can load a specific subset using:

```python
from datasets import load_dataset
dataset = load_dataset("{repo_id}", "subset_name")
```

## Subsets Overview

| Subset Name | Base Dataset | Source Filter | Max Tokens | Samples | Avg Tokens |
|---|---|---|---|---|---|
"""

    configs_metadata = []
    
    for subset_dir in sorted(subsets_path.iterdir()):
        if not subset_dir.is_dir():
            continue
        
        metadata_file = subset_dir / "metadata.json"
        dataset_dir = subset_dir / "dataset"
        
        if not metadata_file.exists() or not dataset_dir.exists():
            print(f"Skipping {subset_dir.name}: missing metadata.json or dataset/ directory.")
            continue
            
        print(f"Processing {subset_dir.name}...")
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            
        # Load dataset
        dataset = load_from_disk(str(dataset_dir))
        
        # Sanitize name for HF config
        config_name = subset_dir.name.replace("openthoughts_all_domains_", "")
        
        print(f"Pushing configuration '{config_name}' to {repo_id}...")
        
        # Force it into a DatasetDict with a 'train' split to ensure it's selectable as 'train'
        if isinstance(dataset, DatasetDict):
            # If it has multiple splits, we might want to keep them or merge them.
            # But the user specifically asked for 'train' set.
            # If it only has 'validation', rename it to 'train'.
            if "validation" in dataset and "train" not in dataset:
                dataset["train"] = dataset.pop("validation")
            elif "train" not in dataset:
                # If no train split at all, take the first one available
                first_split = list(dataset.keys())[0]
                dataset["train"] = dataset.pop(first_split)
        else:
            # Wrap single Dataset into DatasetDict
            dataset = DatasetDict({"train": dataset})
            
        dataset.push_to_hub(repo_id, config_name=config_name)
        
        # Add to YAML configs
        configs_metadata.append({
            "config_name": config_name
        })
        
        # Add to README table
        filters = metadata.get("filters", {})
        stats = metadata.get("stats", {})
        base = metadata.get('base_dataset', 'N/A')
        source = filters.get('source', 'all')
        max_t = filters.get('max_tokens', 'N/A')
        rows = stats.get('total_rows', 'N/A')
        avg_t = stats.get('avg_total_tokens', 0)
        
        readme_content += f"| {config_name} | [{base}](https://huggingface.co/datasets/{base}) | {source} | {max_t} | {rows} | {avg_t:.1f} |\n"

    # Create YAML metadata header
    yaml_metadata = "---\n"
    yaml_metadata += "license: apache-2.0\n"
    yaml_metadata += "task_categories:\n- text-generation\n"
    yaml_metadata += "language:\n- en\n"
    yaml_metadata += "tags:\n- open-thoughts\n- reasoning\n"
    yaml_metadata += "configs:\n"
    for config in configs_metadata:
        yaml_metadata += f"- config_name: {config['config_name']}\n"
    yaml_metadata += "---\n\n"

    full_content = yaml_metadata + readme_content

    # Update README
    with open("TEMP_README.md", "w") as f:
        f.write(full_content)
    
    api.upload_file(
        path_or_fileobj="TEMP_README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset"
    )
    os.remove("TEMP_README.md")
    print("README updated and upload complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Upload dataset subsets to Hugging Face")
    parser.add_argument("repo_id", help="Hugging Face repo ID (e.g., 'username/my-dataset')")
    parser.add_argument("--subsets_dir", default="subsets", help="Directory containing subsets (default: subsets)")
    
    args = parser.parse_args()
    upload_subsets(args.repo_id, args.subsets_dir)

