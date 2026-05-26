from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path

from config import (
    BENCHMARK_MAIN_SET_TOP100,
    BENCHMARK_MAIN_SET_WITH_RUBRICS_TOP100,
    MAX_CONCURRENT_INFERENCE,
    MODEL_SLUG,
    MODEL_TO_EVALUATE,
    TEMPERATURE,
    VISION_DETAIL,
    VLM_ANSWERS_MAIN_TOP100,
    ensure_parent,
    load_project_dotenv,
    resolve_benchmark_image,
)
from openai import AsyncOpenAI
from tqdm import tqdm

MODEL_NAME = MODEL_TO_EVALUATE
MAX_CONCURRENT = MAX_CONCURRENT_INFERENCE
CHECKPOINT_EVERY = max(1, int(os.environ.get("VLM_CHECKPOINT_EVERY", "25")))

PREFERRED_INPUT_JSON = BENCHMARK_MAIN_SET_WITH_RUBRICS_TOP100
FALLBACK_INPUT_JSON = BENCHMARK_MAIN_SET_TOP100
INPUT_JSON = PREFERRED_INPUT_JSON if PREFERRED_INPUT_JSON.exists() else FALLBACK_INPUT_JSON
OUTPUT_JSON = VLM_ANSWERS_MAIN_TOP100

SYSTEM_PROMPT = (
    "You are a precise visual causal reasoning assistant. "
    "Answer the user's constrained question by strictly following every stated "
    "constraint, including forbidden words, coordinate-only output when the question "
    "requires only two numbers (legacy Y2), Coordinate+Explanation two-line output when "
    "the question asks for it (Y2 spatial grounding), contradiction "
    "checks, and JSON-only output when requested. Base your answer on the image."
)


def resolve_image_path(img: dict) -> Path:
    return resolve_benchmark_image(img)


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
    image_url = {"url": f"data:image/jpeg;base64,{base64_image}"}
    if VISION_DETAIL:
        image_url["detail"] = VISION_DETAIL

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question_text},
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
            exc_str = str(exc)
            if "DataInspectionFailed" in exc_str or "inappropriate content" in exc_str:
                print(f"\n[API Blocked] Content filter triggered: {exc_str}")
                return f"[Blocked] API call failed due to safety filter: {exc_str}"
            print(f"\n[API Error] {exc_str}")
            return f"[Error] API call failed: {exc_str}"


def merge_existing_progress(dataset: list[dict], output_json: Path) -> int:
    if not output_json.exists():
        return 0

    with open(output_json, encoding="utf-8") as f:
        existing_data = json.load(f)

    answer_lookup: dict[tuple[int, str, int, int], str] = {}
    for img in existing_data:
        img_id = img.get("image_id")
        for dim, q_list in img.get("questions", {}).items():
            for q_idx, q in enumerate(q_list):
                for var_idx, main in enumerate(q.get("main_set_versions") or []):
                    ans = main.get("vlm_answer", "")
                    if ans and not ans.startswith("[Error]"):
                        answer_lookup[(img_id, dim, q_idx, var_idx)] = ans

    merged = 0
    for img in dataset:
        img_id = img.get("image_id")
        for dim, q_list in img.get("questions", {}).items():
            for q_idx, q in enumerate(q_list):
                for var_idx, main in enumerate(q.get("main_set_versions") or []):
                    key = (img_id, dim, q_idx, var_idx)
                    if key in answer_lookup:
                        main["vlm_answer"] = answer_lookup[key]
                        merged += 1
    return merged


def save_dataset(dataset: list[dict], output_json: Path) -> None:
    """Atomically save progress so interruption does not leave a truncated JSON."""
    ensure_parent(output_json)
    tmp_path = output_json.with_name(output_json.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_json)


async def main() -> None:
    load_project_dotenv()

    print("=" * 70)
    print(f"  Main-Set VLM Inference (Top-100)  |  model: {MODEL_NAME}")
    print("=" * 70)

    if not INPUT_JSON.exists():
        print("[Fatal] Main-set input not found. Run step1b / step1c first.")
        return

    with open(INPUT_JSON, encoding="utf-8") as f:
        dataset: list[dict] = json.load(f)
    print(f"  Input: {INPUT_JSON.name}")

    merged = merge_existing_progress(dataset, OUTPUT_JSON)
    if merged:
        print(f"  Resumed {merged} existing non-error answers from {OUTPUT_JSON.name}.")

    tasks = []
    skipped = 0
    missing_main = 0

    for img in dataset:
        image_path = resolve_image_path(img)
        for dim, q_list in img.get("questions", {}).items():
            for q_idx, q in enumerate(q_list):
                versions = q.get("main_set_versions") or []
                if not versions:
                    missing_main += 1
                    continue
                for var_idx, main_version in enumerate(versions):
                    existing = main_version.get("vlm_answer", "")
                    if existing and not existing.startswith("[Error]"):
                        skipped += 1
                        continue
                    tasks.append({
                        "main_ref": main_version,
                        "image_path": image_path,
                        "question_text": main_version["constrained_question"],
                        "dim": dim,
                        "q_idx": q_idx,
                        "var_idx": var_idx,
                    })

    total_q = sum(
        len(q.get("main_set_versions") or [])
        for img in dataset
        for q_list in img.get("questions", {}).values()
        for q in q_list
    )

    print(f"\n[1/3] {len(dataset)} images | {total_q} constrained variants")
    if missing_main:
        print(f"      Entries missing main_set_versions: {missing_main}")
    if skipped:
        print(f"      Resuming: {skipped} already answered, {len(tasks)} remaining.")

    if tasks:
        print(
            f"\n[2/3] Querying constrained questions "
            f"[concurrency={MAX_CONCURRENT}, checkpoint_every={CHECKPOINT_EVERY}] ..."
        )
        client = AsyncOpenAI()
        sem = asyncio.Semaphore(MAX_CONCURRENT)

        async def run_one(task: dict):
            answer = await ask_vlm(client, sem, task["question_text"], task["image_path"])
            return task, answer

        ok_count = 0
        err_count = 0
        completed = 0
        futures = [asyncio.create_task(run_one(t)) for t in tasks]
        progress = tqdm(total=len(futures), desc="  inference", unit="q")

        try:
            for future in asyncio.as_completed(futures):
                task, answer = await future
                task["main_ref"]["vlm_answer"] = answer
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
                f"\n      [checkpoint] saved current progress "
                f"({completed}/{len(futures)} completed this run) -> {OUTPUT_JSON.name}"
            )

        print(f"\n      Success: {ok_count} | Errors: {err_count}")
    else:
        print("\n[2/3] All constrained variants already answered.")

    print(f"\n[3/3] Saving -> {OUTPUT_JSON.name}")
    save_dataset(dataset, OUTPUT_JSON)

    size_kb = OUTPUT_JSON.stat().st_size / 1024
    print(f"      Saved {len(dataset)} image records ({size_kb:.1f} KB)")
    print("=" * 70)


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
