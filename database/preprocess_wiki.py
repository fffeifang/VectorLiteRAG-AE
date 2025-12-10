import json
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import plyvel
import multiprocessing as mp
import os

from pathlib import Path
PRJ_ROOT = Path(__file__).resolve().parents[1]

NUM_WORKERS = os.cpu_count()
input_path = str(PRJ_ROOT / "database" / "dataset" / "wikidump" / "AA" /"wiki_00")
db_path = str(PRJ_ROOT / "database" / "text_database")

def count_tokens(sentence):
    return len(sentence.split())

def split_sentences(text):
    return [s.strip() for s in sent_tokenize(text) if s.strip() and count_tokens(s.strip()) > 2]

def worker_process(line):
    try:
        obj = json.loads(line)
        text = obj.get("text", "").strip()
        if not text:
            return []
        sentences = split_sentences(text)
        return sentences

    except Exception as e:
        return []

if __name__ == "__main__":
    nltk.download("punkt", quiet=True)

    db = plyvel.DB(db_path, create_if_missing=True)
    with open(input_path, "r", encoding="utf-8") as fin:
        total_lines = sum(1 for _ in fin)

    chunk_id = 0
    with mp.Pool(NUM_WORKERS) as pool, \
         open(input_path, "r", encoding="utf-8") as fin, \
         db.write_batch() as wb:

        for chunks in tqdm(pool.imap(worker_process, fin, chunksize=100),
                           total=total_lines,
                           desc="Processing (MP)"):
            if not chunks:
                continue
            for chunk in chunks:
                if count_tokens(chunk) < 4:
                    continue
                key = f"{chunk_id:012d}".encode("utf-8")
                value = chunk.encode("utf-8")
                wb.put(key, value)
                chunk_id += 1
    db.close()

    print(f"Done. Total chunks = {chunk_id}, saved to LevelDB at: {db_path}")