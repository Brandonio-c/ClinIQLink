import os
import json
import argparse
from pathlib import Path

def split_json_file(input_path, output_dir, chunk_size=50):
    # Load original JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    total = len(data)
    num_chunks = (total + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        chunk = data[i * chunk_size : (i + 1) * chunk_size]
        chunk_path = os.path.join(output_dir, f"mc_chunk_{i+1:03}.json")
        with open(chunk_path, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, indent=2)

    print(f"Split {total} entries into {num_chunks} files in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split MC.json into smaller JSON files of 50 items each.")
    parser.add_argument("--input", required=True, help="Path to the MC.json file")
    parser.add_argument("--output_dir", required=True, help="Directory to save split files")
    parser.add_argument("--chunk_size", type=int, default=50, help="Number of entries per chunk (default=50)")
    args = parser.parse_args()

    split_json_file(args.input, args.output_dir, args.chunk_size)
