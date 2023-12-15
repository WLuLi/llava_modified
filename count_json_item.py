import json
import sys

def count_items_in_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return len(data)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_json_items.py <file>")
        sys.exit(1)

    file = sys.argv[1]
    count = count_items_in_json(file)
    print(f"The number of items in the JSON file is: {count}")
