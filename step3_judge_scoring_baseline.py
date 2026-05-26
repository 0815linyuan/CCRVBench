from __future__ import annotations

import asyncio
import json
import os
from typing import Literal

from config import (
    DIMENSIONS,
    JUDGE_MODEL,
    MAX_CONCURRENT_JUDGE,
    VLM_ANSWERS_BASELINE_TOP100,
    baseline_results_path,
    ensure_parent,
    load_project_dotenv,
)
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

INPUT_JSON = VLM_ANSWERS_BASELINE_TOP100
OUTPUT_JSON = baseline_results_path()
MAX_CONCURRENT = MAX_CONCURRENT_JUDGE


class JudgeScores(BaseModel):
    reasoning: str = Field(description="Brief justification for the scores.")
    r1_score: Literal[0, 1] = Field(description="L1 score 0/1.")
    r2_score: Literal[0, 1] = Field(description="L2 score 0/1.")
    r3_score: Literal[0, 1] = Field(description="L3 score 0/1.")


JUDGE_SYSTEM_PROMPT = """\
You are an automated judge for a Visual Causal Reasoning Benchmark.
Grade the VLM answer against the 3-level rubric. Part-of tolerance applies to L1 only.
Evaluate each level independently; output 1 for Yes, 0 for No.
"""


def compute_dcr(r1: int, r2: int, r3: int) -> tuple[dict, dict, float]:
    r1_prime = r1
    r2_prime = r1 & r2
    r3_prime = r1 & r2 & r3
    dcr = (r1_prime + r2_prime + r3_prime) / 3.0
    raw = {"r1": r1, "r2": r2, "r3": r3}
    cascaded = {"r1_prime": r1_prime, "r2_prime": r2_prime, "r3_prime": r3_prime}
    return raw, cascaded, round(dcr, 3)


async def judge_answer(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    task: dict,
) -> tuple[dict, bool]:
    q_data = task["q_data"]
    vlm_ans = q_data.get("vlm_answer", "")
    rubric = q_data.get("evaluation_rubric", {})

    if not vlm_ans or not rubric:
        return task, False

    user_msg = (
        f'[VLM Answer to Evaluate]:\n"{vlm_ans}"\n\n'
        f"[Rubric]:\n"
        f"L1 (Perception)  : {rubric.get('L1_perception_check', '')}\n"
        f"L2 (Interaction) : {rubric.get('L2_interaction_check', '')}\n"
        f"L3 (Causal)      : {rubric.get('L3_causal_check', '')}"
    )

    async with sem:
        try:
            response = await client.beta.chat.completions.parse(
                model=JUDGE_MODEL,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format=JudgeScores,
            )
            scores: JudgeScores = response.choices[0].message.parsed
            raw, cascaded, dcr = compute_dcr(scores.r1_score, scores.r2_score, scores.r3_score)
            q_data["evaluation_results"] = {
                "raw_scores": raw,
                "cascaded_scores": cascaded,
                "DCR": dcr,
                "judge_reasoning": scores.reasoning,
            }
            return task, True
        except Exception as exc:
            print(f"\n[Error] Judging failed: {exc}")
            return task, False


async def main() -> None:
    load_project_dotenv()

    print("=" * 65)
    print(f"  Judge (Top-100 baseline)  |  judge: {JUDGE_MODEL}")
    print("=" * 65)

    if not INPUT_JSON.exists():
        print(f"[Fatal] {INPUT_JSON.name} not found. Run step2_vlm_inference_baseline.py first.")
        return

    with open(INPUT_JSON, encoding="utf-8") as f:
        dataset: list[dict] = json.load(f)

    tasks: list[dict] = []
    skipped = 0
    for img_entry in dataset:
        for dim_name in DIMENSIONS:
            for q_idx, q_data in enumerate(img_entry.get("questions", {}).get(dim_name, [])):
                if "evaluation_results" in q_data:
                    skipped += 1
                    continue
                tasks.append({
                    "image_entry_ref": img_entry,
                    "dim_name": dim_name,
                    "q_idx": q_idx,
                    "q_data": q_data,
                })

    total_q = sum(
        len(img.get("questions", {}).get(d, []))
        for img in dataset
        for d in DIMENSIONS
    )
    print(f"\n[1/3] {len(dataset)} images  |  {total_q} questions total")
    if skipped:
        print(f"      Resuming: {skipped} already judged, {len(tasks)} remaining.")

    if tasks:
        print(f"[2/3] Judging [concurrency={MAX_CONCURRENT}] …")
        client = AsyncOpenAI()
        sem = asyncio.Semaphore(MAX_CONCURRENT)
        results = await tqdm.gather(
            *(judge_answer(client, sem, t) for t in tasks),
            desc="  judging",
            unit="q",
        )
        ok = sum(1 for _, o in results if o)
        err = sum(1 for _, o in results if not o)
        print(f"\n      Success: {ok}  |  Failed: {err}")
    else:
        print("\n[2/3] All questions already judged.")

    all_results = [
        q_data["evaluation_results"]
        for img in dataset
        for d in DIMENSIONS
        for q_data in img.get("questions", {}).get(d, [])
        if "evaluation_results" in q_data
    ]
    scored = len(all_results)
    print(f"\n[3/3] Scored {scored} / {total_q} questions")
    if scored:
        avg_dcr = sum(r["DCR"] for r in all_results) / scored
        print(f"      Average DCR: {avg_dcr:.3f}")

    ensure_parent(OUTPUT_JSON)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved → {OUTPUT_JSON.name}")
    print("=" * 65)


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
