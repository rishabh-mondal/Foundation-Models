import os
import re
import time
import pandas as pd
from typing import List
from transformers import AutoTokenizer, pipeline

# ---------------------------
# I/O
# ---------------------------
INPUT_CSV  = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/cvpr/combined_vlm_outputs_grouped.csv"
OUTPUT_CSV = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/cvpr/combined_vlm_with_oss_summary.csv"

# ---------------------------
# Primary and fallback models (try in order, per sample)
# Replace or reorder as you wish.
# ---------------------------
MODEL_CANDIDATES = [
    # "openai/gpt-oss-20b",
    # "Qwen/Qwen2.5-7B-Instruct",
    # "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

# Deterministic decoding
MAX_NEW_TOKENS  = 100
DO_SAMPLE       = False
TEMPERATURE     = 0.2   # ignored when DO_SAMPLE=False
TOP_P           = 0.9   # ignored when DO_SAMPLE=False

# Checkpoint interval (write CSV periodically)
CHECKPOINT_EVERY = 10

# Caption columns actually present in your file (no PaliGemma)
CAPTION_COLS = ["InternVL", "Qwen2VL", "GeoChat7B"]

# ---------------------------
# Utilities
# ---------------------------
def clean(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    t = txt.strip().replace('"""', '"')
    t = re.sub(r"\s+", " ", t)
    return t.strip().strip('"').strip("'")

def build_prompt(caps: List[str]) -> str:
    caps_block = "\n".join(f"- {c}" for c in caps if c)
    return (
        f"Captions:\n{caps_block}\n\n"
        "Task: Combine these captions into one concise, factual satellite-image caption describing only visible "
        "objects, land-cover, and location cues (e.g., top-left, center). Avoid meta text or self-reference. "
        "Output ONLY the final caption — no reasoning, no notes, no markdown, no quotes.\n"
        "Final caption:"
    )

def normalize_generated(obj) -> str:
    # HF pipeline may return dict or [dict]
    if isinstance(obj, list):
        gen = obj[0].get("generated_text", "")
    else:
        gen = obj.get("generated_text", "")
    text = gen.strip()
    if "Final caption:" in text:
        text = text.split("Final caption:", 1)[-1].strip()
    text = re.sub(r'["“”]+', '', text).strip()
    text = re.sub(r'(?i)\b(task:|we need|let us|let\'s|instruction)\b.*', '', text).strip()
    words = text.split()
    if len(words) > 100:
        text = " ".join(words[:100])
    return text

def save_csv(names: List[str], paths: List[str], sums: List[str]):
    df_out = pd.DataFrame({
        "image_name": names,
        "image_ptah": paths,        # as requested
        "gpt oss summery": sums     # as requested
    })
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"[Write] {len(df_out)} rows → {OUTPUT_CSV}")

def build_pipeline(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left", trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    gen = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tok,
        torch_dtype="auto",
        device_map="auto",
    )
    return gen

# ---------------------------
# Main (one-by-one, no batching)
# ---------------------------
def main():
    df = pd.read_csv(INPUT_CSV)

    have_cols = [c for c in CAPTION_COLS if c in df.columns]
    if not have_cols:
        raise ValueError("No caption columns found in input CSV.")
    if "image_name" not in df.columns or "path" not in df.columns:
        raise ValueError("Input CSV must contain 'image_name' and 'path'.")

    names, paths, summaries = [], [], []
    sanity_printed = False
    processed = 0
    total = len(df)

    # Build pipelines cache lazily per model
    pipes = {}

    for _, row in df.iterrows():
        image_name = str(row.get("image_name", ""))
        image_path = str(row.get("path", ""))
        caps = [clean(row.get(c, "")) for c in have_cols]
        caps = [c for c in caps if c]

        prompt = build_prompt(caps)
        final_text = ""

        # Try each model until one returns a usable caption
        for mid in MODEL_CANDIDATES:
            try:
                if mid not in pipes:
                    pipes[mid] = build_pipeline(mid)
                out = pipes[mid](
                    prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=DO_SAMPLE,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    return_full_text=False,
                    truncation=True,
                )
                final_text = normalize_generated(out[0])
                if final_text:
                    break
            except Exception as e:
                print(f"[Warn] {mid} failed on {image_name}: {e}")
                continue

        if not final_text:
            final_text = ""  # keep empty; you can insert a placeholder if required

        names.append(image_name)
        paths.append(image_path)
        summaries.append(final_text)

        if not sanity_printed and final_text:
            print("[Sanity] Example output:")
            print(f"image_name={image_name}")
            print(f"image_ptah={image_path}")
            print(f"gpt oss summery={final_text}")
            sanity_printed = True

        processed += 1
        if processed % 50 == 0 or processed == total:
            print(f"[Progress] {processed} / {total}")

        if processed % CHECKPOINT_EVERY == 0 or processed == total:
            save_csv(names, paths, summaries)

    save_csv(names, paths, summaries)
    print("[Done]")

if __name__ == "__main__":
    main()