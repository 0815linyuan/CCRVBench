from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from config import (
    BENCHMARK_MAIN_SET_TOP100,
    BENCHMARK_MAIN_SET_WITH_RUBRICS_TOP100,
    BENCHMARK_WITH_RUBRICS_TOP100,
    PACKAGE_DIR,
    load_project_dotenv,
)

load_project_dotenv()

MODEL = "gpt-5.1"
TEMPERATURE = 0.2
MAX_CONCURRENT = 10
DEFAULT_CHECKPOINT_EVERY = 50
API_TIMEOUT_SEC = 600.0
API_MAX_RETRIES = 2
MAX_INLINE_API_WARNING_PRINTS = 3
DIMENSIONS = ["X1_discovery", "X2_prediction", "X3_diagnosis", "X4_intervention"]
RUBRIC_KEYS = ["L1_perception_check", "L2_interaction_check", "L3_causal_check"]


class EvaluationRubric(BaseModel):
    L1_perception_check: str = Field(
        description="A yes/no check for visual grounding under the constrained answer format."
    )
    L2_interaction_check: str = Field(
        description="A yes/no check for the physical interaction, mechanism, or contradiction detection."
    )
    L3_causal_check: str = Field(
        description="A yes/no check for the final causal resolution under the constrained task."
    )


SYSTEM_PROMPT = """\
You are an expert assessment designer for a constrained Visual Causal Reasoning Benchmark.
Your task is to generate a 3-level DCR rubric for ONE constrained question.

You will receive:
- the original question, ground truth, and rationale
- the original unconstrained L1/L2/L3 rubric when available
- the constrained question
- the adapted constrained ground truth
- the constraint_id and constraint_spec

Generate exactly three binary yes/no checklist questions:

[L1: Perception / Grounding]
[L2: Interaction / Mechanism]
[L3: Causal Resolution]

Important separation:
- DCR rubric evaluates causal reasoning quality.
- CSR separately evaluates constraint compliance.
- Do not output answers to the checklist questions.
- Do not include numeric object IDs.


If constraint_id is "Y2":
- The answer contains both a 'Coordinate' and an 'Explanation'.
- The CSR (Constraint Satisfaction Rate) pipeline will separately and automatically verify if the coordinate is correct.
- Your DCR rubric (L1, L2, L3) MUST focus EXCLUSIVELY on evaluating the textual 'Explanation' part. Formulate your yes/no questions to assess if the physical causality is correctly articulated in the text.
"""


def load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically (temp file + os.replace) to avoid torn files on crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def build_base_rubric_lookup(
    base_data: list[dict],
) -> tuple[dict[tuple[int, str, int], dict], dict[tuple[int, str, str], dict]]:
    index_lookup = {}
    text_lookup = {}
    for img in base_data:
        image_id = img.get("image_id")
        for dim in DIMENSIONS:
            for q_idx, q in enumerate(img.get("questions", {}).get(dim, [])):
                rubric = q.get("evaluation_rubric")
                if rubric:
                    index_lookup[(image_id, dim, q_idx)] = copy.deepcopy(rubric)
                    text_lookup[(image_id, dim, normalize_text(q.get("question_text", "")))] = (
                        copy.deepcopy(rubric)
                    )
    return index_lookup, text_lookup


def validate_rubric(rubric: dict) -> list[str]:
    issues = []
    for key in RUBRIC_KEYS:
        value = rubric.get(key)
        if not isinstance(value, str) or not value.strip():
            issues.append(f"missing_or_empty_{key}")
    return issues


def variant_content_snapshot(main_version: dict) -> tuple[str, str, str]:
    """If this differs from the cached snapshot, step1b (or manual edit) changed the variant."""
    return (
        str(main_version.get("constraint_id", "")),
        str(main_version.get("constrained_question", "")),
        str(main_version.get("adapted_ground_truth", "")),
    )


def main_benchmark_structures_align(main: list, other: list) -> bool:
    """
    Strict alignment so index-based resume is never applied across a reshaped dataset.
    """
    if not isinstance(main, list) or not isinstance(other, list) or not main:
        return False
    if len(main) != len(other):
        return False
    for img_a, img_b in zip(main, other):
        if not isinstance(img_a, dict) or not isinstance(img_b, dict):
            return False
        if img_a.get("image_id") != img_b.get("image_id"):
            return False
        qa = img_a.get("questions")
        qb = img_b.get("questions")
        if not isinstance(qa, dict) or not isinstance(qb, dict):
            return False
        for dim in DIMENSIONS:
            la = qa.get(dim)
            lb = qb.get(dim)
            if not isinstance(la, list) or not isinstance(lb, list):
                return False
            if len(la) != len(lb):
                return False
            for q_a, q_b in zip(la, lb):
                if not isinstance(q_a, dict) or not isinstance(q_b, dict):
                    return False
                va = q_a.get("main_set_versions") or []
                vb = q_b.get("main_set_versions") or []
                if not isinstance(va, list) or not isinstance(vb, list):
                    return False
                if len(va) != len(vb):
                    return False
    return True


def build_resume_cache_from_aligned(
    main: list[dict], existing: list[dict]
) -> dict[tuple[Any, str, int, int], dict[str, Any]]:
    cache: dict[tuple[Any, str, int, int], dict[str, Any]] = {}
    for img_a, img_b in zip(main, existing):
        image_id = img_a.get("image_id")
        qa = img_a["questions"]
        qb = img_b["questions"]
        for dim in DIMENSIONS:
            for q_idx, (q_a, q_b) in enumerate(zip(qa[dim], qb[dim])):
                for var_idx, mv_b in enumerate(q_b.get("main_set_versions") or []):
                    rub = mv_b.get("evaluation_rubric_main_set")
                    if not rub or not isinstance(rub, dict):
                        continue
                    if validate_rubric(rub):
                        continue
                    key = (image_id, dim, q_idx, var_idx)
                    cache[key] = {
                        "rubric": copy.deepcopy(rub),
                        "model": mv_b.get("rubric_generation_model"),
                        "method": mv_b.get("rubric_generation_method"),
                        "snap": variant_content_snapshot(mv_b),
                    }
    return cache


def apply_resume_cache(output_data: list[dict], cache: dict[tuple[Any, str, int, int], dict]) -> int:
    applied = 0
    for img in output_data:
        image_id = img.get("image_id")
        qa = img.get("questions") or {}
        for dim in DIMENSIONS:
            for q_idx, q in enumerate(qa.get(dim) or []):
                for var_idx, main in enumerate(q.get("main_set_versions") or []):
                    key = (image_id, dim, q_idx, var_idx)
                    if key not in cache:
                        continue
                    ent = cache[key]
                    if variant_content_snapshot(main) != ent["snap"]:
                        continue
                    main["evaluation_rubric_main_set"] = copy.deepcopy(ent["rubric"])
                    if ent.get("model") is not None:
                        main["rubric_generation_model"] = ent["model"]
                    if ent.get("method") is not None:
                        main["rubric_generation_method"] = ent["method"]
                    applied += 1
    return applied


def main_version_needs_api(main: dict) -> bool:
    rub = main.get("evaluation_rubric_main_set")
    if not rub or not isinstance(rub, dict):
        return True
    return bool(validate_rubric(rub))


def apply_api_rubric_result(
    output_data: list[dict],
    img_idx: int,
    dim: str,
    q_idx: int,
    var_idx: int,
    rubric: dict[str, str] | None,
    validation_issues: list[dict],
) -> tuple[bool, bool]:
    """
    Write one worker result into output_data. Returns (count_as_ok, count_as_err).
    """
    q = output_data[img_idx]["questions"][dim][q_idx]
    versions = q.get("main_set_versions") or []
    if var_idx >= len(versions):
        return False, True
    main_version = versions[var_idx]
    if rubric:
        main_version["evaluation_rubric_main_set"] = rubric
        main_version["rubric_generation_model"] = MODEL
        main_version["rubric_generation_method"] = "llm_structured_output"
        issues = validate_rubric(rubric)
        if issues:
            validation_issues.append({
                "image_id": output_data[img_idx].get("image_id"),
                "dimension": dim,
                "q_idx": q_idx,
                "var_idx": var_idx,
                "issues": issues,
            })
        return True, False
    return False, True


def collect_tasks(data: list[dict]) -> list[tuple[int, str, int, int, dict, dict]]:
    """(img_idx, dim, q_idx, var_idx, q, main_version_dict)"""
    tasks = []
    for img_idx, img in enumerate(data):
        for dim in DIMENSIONS:
            for q_idx, q in enumerate(img.get("questions", {}).get(dim, [])):
                for var_idx, main in enumerate(q.get("main_set_versions") or []):
                    tasks.append((img_idx, dim, q_idx, var_idx, q, main))
    return tasks


def compact_constraint_spec(spec: dict[str, Any]) -> dict[str, Any]:
    keep_keys = {
        "csr_rule",
        "symbol_map",
        "forbidden_terms",
        "target_true_name",
        "target_bbox_xyxy",
        "target_center_xy",
        "accepted_region",
        "csr_bbox_margin_fraction",
        "original_ground_truth",
        "expected_behavior",
        "false_premise",
        "forbid_extra_text",
        "forbid_punctuation",
        "allowed_max_words",
    }
    return {k: v for k, v in spec.items() if k in keep_keys}


async def generate_main_set_rubric(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    q: dict,
    dim: str,
    main: dict,
    failure_stats: Counter[str],
    failure_samples: list[str],
    inline_warn_remaining: list[int],
) -> dict[str, str] | None:
    user_payload = {
        "x_dimension": dim,
        "original_question": q.get("question_text", ""),
        "original_ground_truth": q.get("ground_truth_answer", ""),
        "original_rationale": q.get("causal_rationale") or q.get("rationale", ""),
        "original_evaluation_rubric": q.get("evaluation_rubric", {}),
        "constraint_id": main.get("constraint_id", ""),
        "constraint_name": main.get("constraint_name", ""),
        "constrained_question": main.get("constrained_question", ""),
        "adapted_ground_truth": main.get("adapted_ground_truth", ""),
        "constraint_spec": compact_constraint_spec(main.get("constraint_spec", {})),
    }

    async with sem:
        try:
            response = await client.beta.chat.completions.parse(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                response_format=EvaluationRubric,
            )
            msg = response.choices[0].message
            parsed: EvaluationRubric | None = msg.parsed
            if parsed is None:
                raise ValueError(
                    "Structured parse returned no payload (parsed=None). "
                    "Check model compatibility with response_format / beta.chat.completions.parse."
                )
            return parsed.model_dump()
        except Exception as exc:
            key = type(exc).__name__
            failure_stats[key] += 1
            if len(failure_samples) < 5:
                failure_samples.append(f"{key}: {exc}")
            if inline_warn_remaining[0] > 0:
                print(f"\n[Warning] Main-set rubric generation failed: {exc}")
                inline_warn_remaining[0] -= 1
            return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate main-set DCR rubrics (resumable).")
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached rubrics in benchmark_main_set_with_rubrics_top100.json and regenerate all variants.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        metavar="N",
        help=(
            f"Atomically save the output JSON every N finished API calls (default: {DEFAULT_CHECKPOINT_EVERY}). "
            "Use 0 to disable periodic checkpoints. On Ctrl+C, progress is still saved once."
        ),
    )
    return p.parse_args()


async def main(force: bool = False, checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY) -> None:
    load_project_dotenv()

    if checkpoint_every < 0:
        print(f"[Warning] --checkpoint-every must be >= 0 (got {checkpoint_every}); using 0.")
        checkpoint_every = 0

    if not BENCHMARK_MAIN_SET_TOP100.exists():
        print(f"[Fatal] {BENCHMARK_MAIN_SET_TOP100.name} not found. Run step1b_generate_main_set.py first.")
        return

    print("=" * 72)
    print("  Main-Set Rubric Generation (Top-100, step1c)")
    if force:
        print("  Mode: --force (full regenerate, no resume)")
    if checkpoint_every > 0:
        print(f"  Checkpoint: every {checkpoint_every} finished API call(s)")
    else:
        print("  Checkpoint: periodic disabled (still saves on normal exit or Ctrl+C)")
    print("=" * 72)
    print(f"\n[1/3] Loading {BENCHMARK_MAIN_SET_TOP100.name} ...")

    output_data: list[dict[str, Any]] = copy.deepcopy(load_json(BENCHMARK_MAIN_SET_TOP100))

    base_index_lookup = {}
    base_text_lookup = {}
    if BENCHMARK_WITH_RUBRICS_TOP100.exists():
        base_index_lookup, base_text_lookup = build_base_rubric_lookup(load_json(BENCHMARK_WITH_RUBRICS_TOP100))
        print(f"      Loaded original rubrics from {BENCHMARK_WITH_RUBRICS_TOP100.name}.")
    else:
        print(f"      [Warning] No base rubric file found at {BENCHMARK_WITH_RUBRICS_TOP100}.")

    copied_original = 0
    for img in output_data:
        image_id = img.get("image_id")
        for dim in DIMENSIONS:
            for q_idx, q in enumerate(img.get("questions", {}).get(dim, [])):
                original_rubric = (
                    base_index_lookup.get((image_id, dim, q_idx))
                    or base_text_lookup.get((image_id, dim, normalize_text(q.get("question_text", ""))))
                    or {}
                )
                if original_rubric:
                    q["evaluation_rubric"] = copy.deepcopy(original_rubric)
                    copied_original += 1

    all_tasks = collect_tasks(output_data)
    print(f"      {len(output_data)} images | {len(all_tasks)} constrained variants")
    print(f"      Original rubrics copied: {copied_original}")

    resume_entries = 0
    reused_applied = 0
    if not force and BENCHMARK_MAIN_SET_WITH_RUBRICS_TOP100.exists():
        out_path = BENCHMARK_MAIN_SET_WITH_RUBRICS_TOP100
        try:
            existing = load_json(out_path)
        except json.JSONDecodeError as exc:
            print(
                f"\n      [Resume] Output file is not valid JSON ({out_path.name}): {exc}\n"
                f"               Falling back to fresh main set only (no rubrics reused)."
            )
        except OSError as exc:
            print(
                f"\n      [Resume] Could not read {out_path.name}: {exc}\n"
                f"               Falling back to fresh main set only (no rubrics reused)."
            )
        else:
            if not isinstance(existing, list):
                print(
                    f"\n      [Resume] Output JSON root is not a list; "
                    f"falling back with no rubrics reused."
                )
            elif not main_benchmark_structures_align(output_data, existing):
                print(
                    f"\n      [Resume] Output file structure does not match "
                    f"{BENCHMARK_MAIN_SET_TOP100.name} (order, counts, or image_id).\n"
                    f"               Ignoring cached rubrics so nothing is silently misaligned."
                )
            else:
                cache = build_resume_cache_from_aligned(output_data, existing)
                resume_entries = len(cache)
                reused_applied = apply_resume_cache(output_data, cache)
                mismatch = resume_entries - reused_applied
                print(
                    f"\n      [Resume] Valid cached rubrics in file: {resume_entries}\n"
                    f"               Reused (snapshot matched current main set): {reused_applied}"
                    + (
                        f"\n               Not reused (variant text changed vs cache): {mismatch}"
                        if mismatch
                        else ""
                    )
                )

    tasks_to_run = [t for t in all_tasks if main_version_needs_api(t[5])]
    out_path = BENCHMARK_MAIN_SET_WITH_RUBRICS_TOP100
    print(f"\n[2/3] Generating constrained rubrics (model={MODEL}, concurrency={MAX_CONCURRENT}) ...")
    print(f"      API calls this run: {len(tasks_to_run)} (skipped: {len(all_tasks) - len(tasks_to_run)})")

    client = AsyncOpenAI(
        timeout=API_TIMEOUT_SEC,
        max_retries=API_MAX_RETRIES,
    )
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    failure_stats: Counter[str] = Counter()
    failure_samples: list[str] = []
    inline_warn_remaining = [MAX_INLINE_API_WARNING_PRINTS]

    async def worker(img_idx: int, dim: str, q_idx: int, var_idx: int, q: dict, main: dict):
        rubric = await generate_main_set_rubric(
            client,
            sem,
            q,
            dim,
            main,
            failure_stats,
            failure_samples,
            inline_warn_remaining,
        )
        return img_idx, dim, q_idx, var_idx, rubric

    ok_count = 0
    err_count = 0
    validation_issues: list[dict] = []
    api_completed = 0

    if tasks_to_run:
        tasks_f = [asyncio.create_task(worker(*t)) for t in tasks_to_run]
        pbar = tqdm(total=len(tasks_f), desc="  main-set rubrics", unit="q")
        try:
            for fut in asyncio.as_completed(tasks_f):
                img_idx, dim, q_idx, var_idx, rubric = await fut
                o, e = apply_api_rubric_result(
                    output_data, img_idx, dim, q_idx, var_idx, rubric, validation_issues
                )
                ok_count += int(o)
                err_count += int(e)
                api_completed += 1
                pbar.update(1)
                if checkpoint_every > 0 and api_completed % checkpoint_every == 0:
                    atomic_write_json(out_path, output_data)
                    tqdm.write(
                        f"      [Checkpoint] {api_completed}/{len(tasks_f)} API calls -> {out_path.name}"
                    )
        except KeyboardInterrupt:
            tqdm.write(f"\n      [Interrupt] Writing partial progress to {out_path.name} ...")
            try:
                atomic_write_json(out_path, output_data)
            except OSError as exc:
                tqdm.write(f"      [Interrupt] Save failed: {exc}")
            if validation_issues:
                try:
                    ip = PACKAGE_DIR / "benchmark_main_set_rubric_issues_top100.json"
                    atomic_write_json(ip, validation_issues)
                    tqdm.write(f"      [Interrupt] Wrote partial issue log -> {ip.name}")
                except OSError as exc:
                    tqdm.write(f"      [Interrupt] Issue log save failed: {exc}")
            tqdm.write(
                f"      [Interrupt] API finished this run: {api_completed} | ok: {ok_count} | failed: {err_count}"
            )
            raise
        finally:
            pbar.close()

    still_missing = sum(1 for t in all_tasks if main_version_needs_api(t[5]))
    print(f"\n      Done - API success: {ok_count} | API failed: {err_count}")
    print(f"      Variants still missing/invalid rubric after this run: {still_missing}")
    print(f"      Validation issues: {len(validation_issues)}")
    if err_count and failure_stats:
        print("\n      API failure counts by exception type:")
        for name, cnt in failure_stats.most_common(12):
            print(f"         {name}: {cnt}")
        print("      Sample messages (up to 5):")
        for s in failure_samples:
            print(f"         - {s}")
        if err_count > MAX_INLINE_API_WARNING_PRINTS:
            print(
                f"      (Only first {MAX_INLINE_API_WARNING_PRINTS} failures printed inline; "
                "see summary above.)"
            )

    print(f"\n[3/3] Saving -> {out_path.name} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, output_data)

    if validation_issues:
        issue_path = PACKAGE_DIR / "benchmark_main_set_rubric_issues_top100.json"
        atomic_write_json(issue_path, validation_issues)
        print(f"      Issue details: {issue_path.name}")

    print("=" * 72)


if __name__ == "__main__":
    _args = parse_args()
    asyncio.run(main(force=_args.force, checkpoint_every=_args.checkpoint_every))
