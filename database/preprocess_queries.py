import csv
import tqdm
import plyvel
import numpy as np

from pathlib import Path
PRJ_ROOT = Path(__file__).resolve().parents[1]

# File paths
tsv_file = str(PRJ_ROOT / "database" / "dataset" / "orcas-doctrain-queries.tsv") # Replace with your TSV file path
db_path = str(PRJ_ROOT / "database" / "query_database")  # LevelDB database path

db = plyvel.DB(db_path, create_if_missing=True)

with open(tsv_file, "r", encoding="utf-8") as file:
    num_lines = sum(1 for line in file) - 1

with open(tsv_file, "r", encoding="utf-8") as file:
    reader = csv.reader(file, delimiter="\t")
    next(reader) 

    with db.write_batch() as batch:
        for index, (qid, query) in enumerate(tqdm.tqdm(reader, total=num_lines, desc="Storing Query TSV to LevelDB")):
            batch.put(str(index).encode(), query.encode())  # Key = index, Value = query

print(f"TSV data successfully stored in LevelDB at {db_path}")
db.close()

