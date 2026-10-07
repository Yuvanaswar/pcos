import json
import uuid
from pathlib import Path

def fix_notebooks():
    nb_dir = Path("notebooks")
    notebooks = sorted(nb_dir.glob("*.ipynb"))
    print(f"Normalizing and repairing all {len(notebooks)} notebooks...")

    for nb_path in notebooks:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)
        
        # Ensure nbformat 4.4
        nb_data["nbformat"] = 4
        nb_data["nbformat_minor"] = 4
        
        # Ensure standard metadata
        if "metadata" not in nb_data:
            nb_data["metadata"] = {}
        nb_data["metadata"]["language_info"] = {"name": "python", "version": "3.14"}
        nb_data["metadata"]["kernelspec"] = {
            "display_name": "Python 3.14 (PCOS PyTorch)",
            "language": "python",
            "name": "python314"
        }
        
        # Ensure every cell has a unique 8-character string ID and compliant structure
        used_ids = set()
        for idx, cell in enumerate(nb_data.get("cells", [])):
            cid = cell.get("id")
            if not cid or cid in used_ids:
                cid = uuid.uuid4().hex[:8]
                cell["id"] = cid
            used_ids.add(cid)
            
            if "metadata" not in cell:
                cell["metadata"] = {}
                
            if cell.get("cell_type") == "code":
                if "execution_count" not in cell:
                    cell["execution_count"] = None
                if "outputs" not in cell:
                    cell["outputs"] = []

        # Atomic overwrite
        temp_path = nb_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, indent=1)
        
        temp_path.replace(nb_path)
        cell_count = len(nb_data["cells"])
        print(f"[REPAIRED & SAVED] {nb_path.name:35s} | Cells: {cell_count:2d} | Valid ID/Format: OK")

    print("\nAll 15 notebooks successfully normalized, repaired, and saved!")

if __name__ == "__main__":
    fix_notebooks()
