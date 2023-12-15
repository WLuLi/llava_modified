import json
import sys

def merge_json_files(file1, file2, output_file):
    merged_data = []

    with open(file1, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
        for item in data1:
            merged_data.append(item)

    with open(file2, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)
        for item in data2:
            merged_data.append(item)

    with open(output_file, 'w', encoding='utf-8') as output:
        json.dump(merged_data, output, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge_json.py <file1> <file2> <output_file>")
        sys.exit(1)

    file1, file2, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    merge_json_files(file1, file2, output_file)
