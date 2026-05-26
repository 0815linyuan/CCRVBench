from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path

from config import (
    BENCHMARK_WITH_RUBRICS_TOP100,
    DIMENSIONS,
    MAX_CONCURRENT_INFERENCE,
    MODEL_TO_EVALUATE,
    TEMPERATURE,
    VISION_DETAIL,
    VLM_ANSWERS_BASELINE_TOP100,
    ensure_parent,
    load_project_dotenv,
    resolve_benchmark_image,
)
from openai import AsyncOpenAI
from tqdm import tqdm


load_project_dotenv()

MODEL_NAME = MODEL_TO_EVALUATE
_max_c_raw = os.environ.get("VLM_MAX_CONCURRENT", "").strip()
MAX_CONCURRENT = (
    int(_max_c_raw) if _max_c_raw else MAX_CONCURRENT_INFERENCE
)
CHECKPOINT_EVERY = max(1, int(os.environ.get("VLM_CHECKPOINT_EVERY", "25")))
# Local OpenAI-compatible VLMs often need hundreds of seconds per vision request.
REQUEST_TIMEOUT_SEC = float(os.environ.get("VLM_REQUEST_TIMEOUT_SECONDS", "600"))
INPUT_JSON = BENCHMARK_WITH_RUBRICS_TOP100

# Choose Variant A (Standard CoT) or Variant B (Contrastive CoT)
COT_VARIANT = "B"

OUTPUT_JSON = VLM_ANSWERS_BASELINE_TOP100.with_name(
    VLM_ANSWERS_BASELINE_TOP100.name.replace(".json", f"_cot_{COT_VARIANT}.json")
)

SYSTEM_PROMPT = (
    "You are a precise visual reasoning assistant evaluating causal relationships "
    "in images. When answering, always:\n"
    "1. Identify the relevant physical entities in the image.\n"
    "2. Describe the physical action or contact between them.\n"
    "3. State the causal outcome directly and concisely.\n"
    "Do NOT refuse to answer or say you cannot see the image. "
    "Base your answer strictly on what is visually present."
)


def encode_image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def ask_vlm(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    question_text: str,
    image_path: Path,
) -> str:
    if not image_path.exists():
        return f"[Error] Image not found: {image_path}"

    base64_image = encode_image_to_base64(image_path)
    image_url: dict = {"url": f"data:image/jpeg;base64,{base64_image}"}
    if VISION_DETAIL:
        image_url["detail"] = VISION_DETAIL

    cot_prompt_A = "Please think step by step before providing your final concise answer."
    cot_prompt_B = (
        "First, think step by step. Consider what the most plausible incorrect assumption might be in this scenario. "
        "Briefly refute that incorrect assumption based on visual evidence, and then provide your final concise answer."
    )
    cot_prompt = cot_prompt_A if COT_VARIANT == "A" else cot_prompt_B

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Look at the image carefully and answer the following "
                        "causal reasoning question. Be physically precise.\n"
                        f"{cot_prompt}\n\n"
                        f"Question: {question_text}"
                    ),
                },
                {"type": "image_url", "image_url": image_url},
            ],
        },
    ]

    async with sem:
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"\n[API Error] {exc}")
            return f"[Error] API call failed: {exc}"


def save_dataset(dataset: list[dict], output_json: Path) -> None:
    """Atomic write so interruption during save does not leave truncated JSON."""
    ensure_parent(output_json)
    tmp_path = output_json.with_name(output_json.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_json)


async def main() -> None:
    load_project_dotenv()

    print("=" * 65)
    print(f"  VLM Inference (Top-100 baseline)  |  model: {MODEL_NAME}")
    print(
        f"  timeout={REQUEST_TIMEOUT_SEC}s/request  |  "
        f"concurrency={MAX_CONCURRENT}  (override with env if needed)"
    )
    print("=" * 65)

    if not INPUT_JSON.exists():
        print(f"[Fatal] {INPUT_JSON.name} not found. Run step1_generate_rubrics_baseline.py first.")
        return

    with open(INPUT_JSON, encoding="utf-8") as f:
        dataset: list[dict] = json.load(f)

    if OUTPUT_JSON.exists():
        print(f"  Found existing output for resume: {OUTPUT_JSON.name}")
        with open(OUTPUT_JSON, encoding="utf-8") as f:
            existing_data = json.load(f)
        existing_answers: dict[tuple[int, str, object], str] = {}
        for img in existing_data:
            img_id = img.get("image_id")
            for dim in DIMENSIONS:
                for q in img.get("questions", {}).get(dim, []):
                    ans = q.get("vlm_answer", "")
                    if ans and not ans.startswith("[Error]"):
                        q_key = q.get("target_object_id", q.get("question_text"))
                        existing_answers[(img_id, dim, q_key)] = ans

        for img_entry in dataset:
            img_id = img_entry.get("image_id")
            for dim_name in DIMENSIONS:
                for q_data in img_entry.get("questions", {}).get(dim_name, []):
                    q_key = q_data.get("target_object_id", q_data.get("question_text"))
                    if (img_id, dim_name, q_key) in existing_answers:
                        q_data["vlm_answer"] = existing_answers[(img_id, dim_name, q_key)]

    tasks: list[dict] = []
    skipped = 0

    for img_entry in dataset:
        image_path = resolve_benchmark_image(img_entry)
        for dim_name in DIMENSIONS:
            for q_idx, q_data in enumerate(img_entry.get("questions", {}).get(dim_name, [])):
                existing = q_data.get("vlm_answer", "")
                if existing and not existing.startswith("[Error]"):
                    skipped += 1
                    continue
                tasks.append({
                    "image_entry_ref": img_entry,
                    "dim_name": dim_name,
                    "q_idx": q_idx,
                    "image_path": image_path,
                    "question_text": q_data["question_text"],
                })

    total_q = sum(
        len(img.get("questions", {}).get(d, []))
        for img in dataset
        for d in DIMENSIONS
    )
    print(f"\n[1/3] {len(dataset)} images  |  {total_q} questions total")
    if skipped:
        print(f"      Resuming: {skipped} already answered, {len(tasks)} remaining.")

    if not tasks:
        print("  All questions already answered.")
    else:
        print(
            f"[2/3] Querying VLM [concurrency={MAX_CONCURRENT}, "
            f"checkpoint_every={CHECKPOINT_EVERY}] …"
        )
        client = AsyncOpenAI(timeout=REQUEST_TIMEOUT_SEC)
        sem = asyncio.Semaphore(MAX_CONCURRENT)

        async def run_one(task: dict):
            answer = await ask_vlm(client, sem, task["question_text"], task["image_path"])
            return task, answer

        futures = [asyncio.create_task(run_one(t)) for t in tasks]
        ok_count = 0
        err_count = 0
        completed = 0
        progress = tqdm(total=len(futures), desc="  inference", unit="q")

        try:
            for fut in asyncio.as_completed(futures):
                try:
                    task, answer = await fut
                except asyncio.CancelledError:
                    completed += 1
                    progress.update(1)
                    continue

                q_obj = task["image_entry_ref"]["questions"][task["dim_name"]][task["q_idx"]]
                q_obj["vlm_answer"] = answer
                if answer.startswith("[Error]"):
                    err_count += 1
                else:
                    ok_count += 1

                completed += 1
                progress.update(1)

                if completed % CHECKPOINT_EVERY == 0:
                    save_dataset(dataset, OUTPUT_JSON)
                    progress.write(
                        f"      [checkpoint] saved {completed}/{len(futures)} "
                        f"completed -> {OUTPUT_JSON.name}"
                    )
        finally:
            progress.close()
            for future in futures:
                if not future.done():
                    future.cancel()
            save_dataset(dataset, OUTPUT_JSON)
            print(
                f"\n      [checkpoint] flushed progress "
                f"({completed}/{len(futures)} completed this run) -> {OUTPUT_JSON.name}"
            )

        print(f"\n      Success: {ok_count}  |  Errors: {err_count}")

    print(f"\n[3/3] Saving → {OUTPUT_JSON.name} …")
    save_dataset(dataset, OUTPUT_JSON)
    print(f"      Saved ({OUTPUT_JSON.stat().st_size / 1024:.1f} KB)")
    print("=" * 65)


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
