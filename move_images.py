import os
import shutil
import sys

def copy_missing_files(missing_files_list, source_root, destination_root):
    with open(missing_files_list, 'r') as file:
        for line in file:
            source_file = os.path.join(source_root, line.strip())
            destination_file = os.path.join(destination_root, line.strip())

            os.makedirs(os.path.dirname(destination_file), exist_ok=True)

            if os.path.exists(source_file):
                shutil.copy2(source_file, destination_file)
                print(f"Copied {source_file} to {destination_file}")
            else:
                print(f"Source file not found: {source_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python copy_missing_files.py <missing_files_list> <source_root> <destination_root>")
        sys.exit(1)

    missing_files_list = sys.argv[1]
    source_root = sys.argv[2]
    destination_root = sys.argv[3]

    copy_missing_files(missing_files_list, source_root, destination_root)
