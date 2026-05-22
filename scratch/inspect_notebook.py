import json

with open("train_board_segmentation_colab-new.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    print(f"--- Cell {idx} ({cell['cell_type']}) ---")
    source = cell.get("source", [])
    snippet = "".join(source[:5]) if source else "[Empty]"
    print(snippet)
    print("-----------------------\n")
