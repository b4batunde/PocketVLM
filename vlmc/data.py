"""Dataset preparation: one materialized copy of train / calib / eval that every backend reads.

Items are JSONL rows: {qid, image, question, answers[10], answer (consensus), answer_type}.
Images are written once as JPEG so HF, llm-compressor and llama.cpp see identical pixels.

VQAv2 only ships answers for train/val, and the HF mirrors of train are script-based, so we
partition the *validation* split by image_id: eval images never appear in train or calib.
Numbers are therefore comparable across our variants, not to the official test-dev leaderboard.
"""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def _image_bucket(image_id) -> float:
    h = hashlib.sha1(str(image_id).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _answers(row) -> list[str]:
    a = row.get("answers") or []
    out = [x["answer"] if isinstance(x, dict) else str(x) for x in a]
    return out


def _consensus(row, answers):
    mc = row.get("multiple_choice_answer")
    if mc:
        return mc
    return Counter(answers).most_common(1)[0][0] if answers else ""


def prepare_vqav2(cfg: dict, out_dir: Path) -> dict:
    from datasets import load_dataset

    d = cfg["data"]
    rng = random.Random(d["seed"])
    # Only download the split we use. lmms-lab/VQAv2 also ships test/testdev (~3x the size, with image
    # bytes embedded per question), which otherwise land in the HF cache too.
    try:
        ds = load_dataset(d["hf_id"], data_files={d["split"]: f"data/{d['split']}-*.parquet"}, split=d["split"])
    except Exception as e:  # different repo layout: fall back to the whole dataset
        print(f"[data] split-only load failed ({e}); loading the full dataset")
        ds = load_dataset(d["hf_id"], split=d["split"])
    image_ids = ds["image_id"]
    eval_idx, train_idx = [], []
    for i, img in enumerate(image_ids):
        (eval_idx if _image_bucket(img) < d["eval_image_frac"] else train_idx).append(i)
    rng.shuffle(eval_idx)
    rng.shuffle(train_idx)
    splits = {
        "eval": eval_idx[: d["eval_questions"]],
        "train": train_idx[: d["train_questions"]],
        "calib": train_idx[d["train_questions"]: d["train_questions"] + d["calib_questions"]],
    }
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, idx in splits.items():
        sub = ds.select(sorted(idx))
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for row in sub:
                p = img_dir / f"{row['image_id']}.jpg"
                if not p.exists():
                    row["image"].convert("RGB").save(p, quality=95)
                ans = _answers(row)
                f.write(json.dumps({
                    "qid": str(row.get("question_id")), "image": str(p.relative_to(out_dir)),
                    "question": row["question"], "answers": ans, "answer": _consensus(row, ans),
                    "answer_type": row.get("answer_type", "unknown"),
                }) + "\n")
        stats[name] = len(idx)
    # sanity: image-level disjointness
    def imgs(n):
        return {json.loads(l)["image"] for l in open(out_dir / f"{n}.jsonl")}
    assert not (imgs("eval") & (imgs("train") | imgs("calib"))), "eval images leaked into train/calib"
    stats["source"] = f"{d['hf_id']}:{d['split']}"
    json.dump(stats, open(out_dir / "splits.json", "w"), indent=2)
    return stats


def prepare_synthetic(out_dir: Path, n_train=64, n_calib=16, n_eval=32, seed=0) -> dict:
    """Shapes-and-colors VQA for smoke tests (no network)."""
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    colors = {"red": (220, 30, 30), "blue": (30, 60, 220), "green": (30, 170, 60), "yellow": (230, 210, 40)}
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    stats, k = {}, 0
    for name, n in [("train", n_train), ("calib", n_calib), ("eval", n_eval)]:
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for _ in range(n):
                cname, rgb = rng.choice(list(colors.items()))
                count = rng.randint(1, 4)
                im = Image.new("RGB", (112, 112), "white")
                dr = ImageDraw.Draw(im)
                for c in range(count):
                    x = 8 + c * 25
                    dr.rectangle([x, 40, x + 18, 58], fill=rgb)
                p = img_dir / f"{name}_{k}.jpg"
                im.save(p)
                if rng.random() < 0.5:
                    q, a, t = "What color are the squares?", cname, "other"
                else:
                    q, a, t = "How many squares are there?", str(count), "number"
                f.write(json.dumps({"qid": str(k), "image": str(p.relative_to(out_dir)), "question": q,
                                    "answers": [a] * 10, "answer": a, "answer_type": t}) + "\n")
                k += 1
        stats[name] = n
    stats["source"] = "synthetic"
    json.dump(stats, open(out_dir / "splits.json", "w"), indent=2)
    return stats


def load_items(path: str | Path, limit: int | None = None) -> list[dict]:
    path = Path(path)
    items = [json.loads(l) for l in open(path)]
    for it in items:
        it["image_path"] = str(path.parent / it["image"])
    return items[:limit] if limit else items
