from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Literal

from config import (
    BENCHMARK_MAIN_SET_WITH_RUBRICS_TOP100,
    JUDGE_MODEL,
    MAX_CONCURRENT_JUDGE,
    MODEL_TO_EVALUATE,
    MODEL_SLUG,
    RESULTS_MAIN_TOP100,
    VLM_ANSWERS_MAIN_TOP100,
    ensure_parent,
    load_project_dotenv,
)
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm
from y4_word_normalize import normalize_y4_words, y4_has_forbidden_punctuation

INPUT_JSON = VLM_ANSWERS_MAIN_TOP100
OUTPUT_JSON = RESULTS_MAIN_TOP100
RUBRIC_SOURCE_JSON = BENCHMARK_MAIN_SET_WITH_RUBRICS_TOP100
MAX_CONCURRENT = MAX_CONCURRENT_JUDGE


class DcrJudgeScores(BaseModel):
    reasoning: str = Field(description="Brief justification for the three DCR scores.")
    r1_score: Literal[0, 1] = Field(description="L1 visual entity/perception score.")
    r2_score: Literal[0, 1] = Field(description="L2 interaction/mechanism score.")
    r3_score: Literal[0, 1] = Field(description="L3 causal outcome score.")


class CsrJudgeScores(BaseModel):
    reasoning: str = Field(description="Brief justification for the CSR decision.")
    csr_score: Literal[0, 1] = Field(description="1 if the answer follows the constraint, otherwise 0.")


DCR_JUDGE_PROMPT = """\
You are a strict but fair judge for a constrained Visual Causal Reasoning benchmark.
Score the VLM answer against the provided constrained L1/L2/L3 rubric.
Score causal quality here, not stylistic compliance; constraint compliance is
judged separately as CSR.

Use independent binary scores:
- L1 perception: the answer identifies or grounds the relevant visual entity/entities.
  Part-of tolerance applies here only: a visually relevant part of a target entity
  can satisfy L1, e.g. person -> hand/arm/leg/head, vehicle -> wheel/door.
- L2 interaction/mechanism: the answer captures the relevant physical interaction,
  spatial relation, intervention target, or causal mechanism.
- L3 causal outcome: the answer reaches the correct causal consequence, prediction,
  diagnosis, or intervention effect.

For Y2 (spatial grounding), the answer must include BOTH a Coordinate line and an Explanation line.
Grade causal quality primarily from the Explanation (mechanism and outcome). The Coordinate line
supports grounding only.

For legacy coordinate-only Y2 answers (two numbers only, no Explanation line), L1/L2/L3 may still
be satisfied when the coordinate correctly denotes the causal target; do not penalize missing prose
if the constrained question required no prose.

Output only the structured scores.
"""

CSR_Y3_PROMPT = """\
You judge whether a VLM complied with a Y3 contradiction-blocking constraint.
CSR is about constraint following, not whether the final causal answer is otherwise good.

Y3 is NOT a counterfactual reasoning task.
The textual premise must be treated as a claim that may be false and must be checked
against the image/visual causal evidence. The model should not accept the premise
as a hypothetical world unless the question explicitly says to do counterfactual
reasoning, which Y3 does not.

Give CSR=1 only if the answer does all of the following:
1. Indicates that the textual premise is false, questionable, inconsistent, or
   contradicted by the visual evidence.
2. Refuses, corrects, or does not reason from the false premise as if it were true.
3. Bases the response on the image/visual evidence after rejecting or correcting
   the false premise.

Give CSR=0 if any of the following is true:
- The answer blindly accepts the false premise.
- The answer treats the false premise as a counterfactual/hypothetical assumption
  and continues the requested inference from that assumption.
- The answer gives a causal continuation without first flagging the contradiction.
- The answer is ambiguous and does not clearly resist or correct the false premise.

Do not require a specific phrase such as "I refuse"; semantically equivalent
contradiction detection and correction is sufficient.
"""


def compute_dcr(r1: int, r2: int, r3: int) -> tuple[dict, dict, float]:
    r1_prime = r1
    r2_prime = r1 & r2
    r3_prime = r1 & r2 & r3
    dcr = (r1_prime + r2_prime + r3_prime) / 3.0
    raw = {"r1": r1, "r2": r2, "r3": r3}
    cascaded = {"r1_prime": r1_prime, "r2_prime": r2_prime, "r3_prime": r3_prime}
    return raw, cascaded, round(dcr, 3)


def build_main_set_rubric_lookup(path) -> dict[tuple[int, str, int, int], dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        dataset = json.load(f)

    lookup: dict[tuple[int, str, int, int], dict] = {}
    for img in dataset:
        image_id = img.get("image_id")
        for dim, q_list in img.get("questions", {}).items():
            for q_idx, q in enumerate(q_list):
                for var_idx, main in enumerate(q.get("main_set_versions") or []):
                    rubric = main.get("evaluation_rubric_main_set")
                    if rubric:
                        lookup[(image_id, dim, q_idx, var_idx)] = rubric
    return lookup


def merge_missing_main_set_rubrics(
    dataset: list[dict], rubric_lookup: dict[tuple[int, str, int, int], dict]
) -> int:
    merged = 0
    for img in dataset:
        image_id = img.get("image_id")
        for dim, q_list in img.get("questions", {}).items():
            for q_idx, q in enumerate(q_list):
                for var_idx, main in enumerate(q.get("main_set_versions") or []):
                    if main.get("evaluation_rubric_main_set"):
                        continue
                    rubric = rubric_lookup.get((image_id, dim, q_idx, var_idx))
                    if rubric:
                        main["evaluation_rubric_main_set"] = rubric
                        merged += 1
    return merged


def normalize_constraint_id(constraint_id: str) -> str:
    match = re.match(r"^(Y\d+\.\d+)", constraint_id or "")
    return match.group(1) if match else constraint_id


def forbidden_terms_used(answer: str, terms: list[str]) -> list[str]:
    lowered = answer.lower()
    used = []
    for term in terms:
        clean = str(term).strip().lower()
        if not clean:
            continue
        pattern = r"(?<![a-z0-9_])" + re.escape(clean) + r"(?![a-z0-9_])"
        if re.search(pattern, lowered):
            used.append(term)
    return used


def parse_y2_coordinate_pair_strict(answer: str) -> tuple[float, float] | None:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer)
    if len(numbers) != 2:
        return None
    return float(numbers[0]), float(numbers[1])


Y2_EXPLANATION_MIN_CHARS = 20
Y2_DEFAULT_BBOX_MARGIN_FRAC = 0.075


def expand_bbox_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    margin_fraction: float,
) -> tuple[float, float, float, float]:
    """Outward-expand axis-aligned box by margin_fraction of its width/height per side."""
    lo_x, hi_x = (x1, x2) if x1 <= x2 else (x2, x1)
    lo_y, hi_y = (y1, y2) if y1 <= y2 else (y2, y1)
    w = hi_x - lo_x
    h = hi_y - lo_y
    dx = w * margin_fraction
    dy = h * margin_fraction
    return lo_x - dx, lo_y - dy, hi_x + dx, hi_y + dy


def bbox_xyxy_tuple_from_spec(spec: dict) -> tuple[float, float, float, float] | None:
    if not spec:
        return None
    raw = spec.get("target_bbox_xyxy")
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        return float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])
    meta = spec.get("target_bbox") or spec.get("target_bbox_metadata")
    if isinstance(meta, dict) and "xyxy" in meta:
        t = meta["xyxy"]
        if isinstance(t, (list, tuple)) and len(t) == 4:
            return float(t[0]), float(t[1]), float(t[2]), float(t[3])
    return None


def parse_y2_grounded_format(answer: str) -> tuple[tuple[float, float] | None, str | None, str | None]:
    """
    Parse Y2 dual-line format. Returns (point, explanation_text, error_reason).
    explanation_text is trimmed content after 'Explanation:' (first line); continuation lines may exist but are optional for CSR.
    """
    text = (answer or "").strip()
    if not text:
        return None, None, "Empty answer."

    lines = [ln.rstrip() for ln in text.splitlines()]
    coord_line = next((ln for ln in lines if ln.strip().lower().startswith("coordinate:")), None)
    expl_line = next((ln for ln in lines if ln.strip().lower().startswith("explanation:")), None)
    if not coord_line or not expl_line:
        return None, None, None  # not grounded format

    _, coord_rest = coord_line.split(":", 1)
    point = parse_y2_coordinate_pair_strict(coord_rest)
    if point is None:
        return None, None, "Coordinate line must contain exactly two numeric values."

    _, expl_rest = expl_line.split(":", 1)
    explanation = expl_rest.strip()
    # Include following lines if model wrapped Explanation across N lines (first line after colon short)
    expl_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("explanation:"):
            expl_idx = i
            break
    if expl_idx is not None:
        tail = [lines[expl_idx].split(":", 1)[1].strip()] if ":" in lines[expl_idx] else []
        for j in range(expl_idx + 1, len(lines)):
            low = lines[j].strip().lower()
            if low.startswith("coordinate:"):
                break
            if lines[j].strip():
                tail.append(lines[j].strip())
        explanation = " ".join(t for t in tail if t).strip()

    if len(explanation) < Y2_EXPLANATION_MIN_CHARS:
        return None, None, f"Explanation must be at least {Y2_EXPLANATION_MIN_CHARS} characters."

    return point, explanation, None


Y2_ALLOWED_ENGLISH_WORDS = frozenset({
    "x", "y", "xy", "coord", "coords", "coordinate", "coordinates", "point", "points",
    "center", "centre", "approx", "approximately", "approximate", "px", "pixel", "pixels",
    "mm", "cm", "m",
})


def point_inside_bbox(
    point: tuple[float, float],
    bbox: dict,
    *,
    margin_fraction: float | None = None,
) -> bool:
    x, y = point
    if margin_fraction is not None and margin_fraction > 0:
        if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            coords = expand_bbox_xyxy(
                float(bbox["x1"]),
                float(bbox["y1"]),
                float(bbox["x2"]),
                float(bbox["y2"]),
                margin_fraction,
            )
            x1, y1, x2, y2 = coords
        elif "xyxy" in bbox:
            x1, y1, x2, y2 = expand_bbox_xyxy(
                *bbox["xyxy"],
                margin_fraction,
            )
        elif "target_bbox_xyxy" in bbox:
            b = bbox["target_bbox_xyxy"]
            x1, y1, x2, y2 = expand_bbox_xyxy(float(b[0]), float(b[1]), float(b[2]), float(b[3]), margin_fraction)
        elif "xywh" in bbox:
            bx, by, bw, bh = bbox["xywh"]
            x1, y1, x2, y2 = expand_bbox_xyxy(bx, by, bx + bw, by + bh, margin_fraction)
        elif "target_bbox_xywh" in bbox:
            bx, by, bw, bh = bbox["target_bbox_xywh"]
            x1, y1, x2, y2 = expand_bbox_xyxy(bx, by, bx + bw, by + bh, margin_fraction)
        else:
            return False
    elif all(k in bbox for k in ("x1", "y1", "x2", "y2")):
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    elif "xyxy" in bbox:
        x1, y1, x2, y2 = bbox["xyxy"]
    elif "xywh" in bbox:
        bx, by, bw, bh = bbox["xywh"]
        x1, y1, x2, y2 = bx, by, bx + bw, by + bh
    elif "target_bbox_xyxy" in bbox:
        x1, y1, x2, y2 = bbox["target_bbox_xyxy"]
    elif "target_bbox_xywh" in bbox:
        bx, by, bw, bh = bbox["target_bbox_xywh"]
        x1, y1, x2, y2 = bx, by, bx + bw, by + bh
    else:
        return False
    lo_x, hi_x = (x1, x2) if x1 <= x2 else (x2, x1)
    lo_y, hi_y = (y1, y2) if y1 <= y2 else (y2, y1)
    return float(lo_x) <= x <= float(hi_x) and float(lo_y) <= y <= float(hi_y)


def y2_coordinate_only_text_violation(answer: str, spec: dict) -> tuple[bool, str]:
    if re.search(r"[\u4e00-\u9fff]", answer):
        return True, "Chinese text is not allowed in coordinate-only output."

    leak_terms: list[str] = []
    target_name = spec.get("target_true_name")
    if isinstance(target_name, str) and target_name.strip():
        leak_terms.append(target_name.strip())

    used = forbidden_terms_used(answer, leak_terms)
    if used:
        return True, f"Entity/object name leak: {used}"

    words = re.findall(r"[A-Za-z]+", answer.lower())
    bad = [w for w in words if w not in Y2_ALLOWED_ENGLISH_WORDS]
    if bad:
        preview = bad[:12]
        suffix = "..." if len(bad) > 12 else ""
        return True, f"Disallowed non-format words: {preview}{suffix}"

    return False, ""


def check_y1_csr(answer: str, spec: dict) -> tuple[int, str]:
    terms = spec.get("forbidden_terms", [])
    used = forbidden_terms_used(answer, terms)
    if used:
        return 0, f"Forbidden real entity terms used: {used}"
    return 1, "No forbidden real entity terms detected."


def check_y2_csr(answer: str, spec: dict) -> tuple[int, str]:
    margin_raw = spec.get("csr_bbox_margin_fraction", Y2_DEFAULT_BBOX_MARGIN_FRAC)
    try:
        margin = float(margin_raw)
    except (TypeError, ValueError):
        margin = Y2_DEFAULT_BBOX_MARGIN_FRAC
    margin = max(0.0, min(margin, 0.25))

    xyxy = bbox_xyxy_tuple_from_spec(spec)
    target_bbox = spec.get("target_bbox") or spec.get("target_bbox_metadata") or spec

    def point_in_target_region(point: tuple[float, float]) -> bool:
        if xyxy is not None:
            ex = expand_bbox_xyxy(*xyxy, margin)
            inner = {"x1": ex[0], "y1": ex[1], "x2": ex[2], "y2": ex[3]}
            return point_inside_bbox(point, inner)
        return point_inside_bbox(point, target_bbox, margin_fraction=margin)

    def has_bbox_for_csr() -> bool:
        if xyxy is not None:
            return True
        if isinstance(target_bbox, dict) and any(
            k in target_bbox for k in ("target_bbox_xyxy", "x1", "xyxy", "xywh", "target_bbox_xywh")
        ):
            return True
        return False

    grounded_point, _expl, g_err = parse_y2_grounded_format(answer)
    if g_err is not None:
        return 0, g_err

    if spec.get("csr_rule") == "grounded_causal_explanation":
        if grounded_point is None or _expl is None:
            return (
                0,
                "Y2 'grounded_causal_explanation' strictly requires 'Coordinate:' and 'Explanation:' lines. Legacy coordinate-only answer is rejected."
            )
    if grounded_point is not None and _expl is not None:
        if not has_bbox_for_csr():
            return 0, "No target bbox metadata is available for coordinate validation."
        if not point_in_target_region(grounded_point):
            return (
                0,
                f"Coordinate {grounded_point} is outside the expanded target bbox (margin={margin:.3f}).",
            )
        return 1, (
            f"Y2 grounded format ok; coordinate {grounded_point} inside expanded bbox "
            f"(margin fraction {margin:.3f})."
        )

    lowered = (answer or "").lower()
    if "coordinate:" in lowered or "explanation:" in lowered:
        return (
            0,
            "Incomplete Y2 grounded format: require both 'Coordinate:' and 'Explanation:' lines "
            f"(Explanation >= {Y2_EXPLANATION_MIN_CHARS} characters).",
        )

    point = parse_y2_coordinate_pair_strict(answer)
    if point is None:
        return (
            0,
            "Y2 requires either (1) Coordinate: + Explanation: lines or (2) legacy: exactly two numbers total.",
        )

    if not has_bbox_for_csr():
        return 0, "No target bbox metadata is available for coordinate validation."

    if not point_in_target_region(point):
        return 0, f"Coordinate {point} is outside the expanded target bbox (margin={margin:.3f})."

    bad_text, reason = y2_coordinate_only_text_violation(answer, spec)
    if bad_text:
        return 0, reason
    return 1, f"Legacy Y2 ok: coordinate {point} inside expanded bbox (margin fraction {margin:.3f})."


def check_y4_json_schema_csr(answer: str, spec: dict) -> tuple[int, str]:
    """Legacy Y4: single JSON object with causal_answer + causal_confidence."""
    stripped = answer.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return 0, "Answer is not a JSON object only."
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return 0, f"Invalid JSON: {exc}"
    if not isinstance(parsed, dict):
        return 0, "JSON root is not an object."

    required_keys = set(spec.get("required_keys") or spec.get("required_schema", {}).keys())
    if not required_keys:
        required_keys = {"causal_answer", "causal_confidence"}
    if spec.get("forbid_extra_text", True) and set(parsed.keys()) != required_keys:
        return 0, f"JSON keys must be exactly {sorted(required_keys)}."
    if not required_keys.issubset(parsed.keys()):
        return 0, f"JSON is missing required keys: {sorted(required_keys - set(parsed.keys()))}."

    schema = spec.get("required_schema", {})
    for key, expected_type in schema.items():
        value = parsed.get(key)
        if expected_type == "string" and not isinstance(value, str):
            return 0, f"{key} must be a string."
        if expected_type == "number_between_0_and_1":
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                return 0, f"{key} must be a number from 0.0 to 1.0."
    return 1, "Valid strict JSON object with required schema."


def check_y4_dynamic_word_limit_csr(answer: str, spec: dict) -> tuple[int, str]:
    stripped = (answer or "").strip()
    if not stripped:
        return 0, "Y4 answer is empty."
    max_words = spec.get("allowed_max_words")
    if max_words is None or not isinstance(max_words, int) or max_words < 1:
        return 0, "Y4 spec missing valid allowed_max_words."

    if spec.get("forbid_extra_text", True) and "\n" in stripped:
        return 0, "Y4 requires a single-line answer (no newlines)."

    words = normalize_y4_words(stripped)
    if not words:
        return 0, "Y4 answer has no words after normalization."
    if len(words) > max_words:
        return 0, f"Y4 word count {len(words)} exceeds limit {max_words}."

    if spec.get("forbid_punctuation", True):
        if y4_has_forbidden_punctuation(stripped):
            return 0, "Y4 forbids punctuation in the answer (hyphens in compounds are allowed)."

    return 1, f"Y4 ok: {len(words)} words (max {max_words})."


def check_y4_csr(answer: str, spec: dict) -> tuple[int, str]:
    rule = spec.get("csr_rule")
    if rule == "dynamic_word_limit":
        return check_y4_dynamic_word_limit_csr(answer, spec)
    if rule == "json_schema" or spec.get("required_schema"):
        return check_y4_json_schema_csr(answer, spec)
    return check_y4_dynamic_word_limit_csr(answer, spec)


async def judge_y3_csr(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    answer: str,
    question_text: str,
    spec: dict,
) -> tuple[int, str]:
    user_msg = (
        f"[Constraint Spec]\n{json.dumps(spec, ensure_ascii=False, indent=2)}\n\n"
        f"[Constrained Question]\n{question_text}\n\n"
        f"[VLM Answer]\n{answer}"
    )
    async with sem:
        response = await client.beta.chat.completions.parse(
            model=JUDGE_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": CSR_Y3_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format=CsrJudgeScores,
        )
    scores: CsrJudgeScores = response.choices[0].message.parsed
    return scores.csr_score, scores.reasoning


def check_rule_based_csr(answer: str, constraint_id: str, spec: dict) -> tuple[int, str] | None:
    y_id = normalize_constraint_id(constraint_id)
    if y_id == "Y1":
        return check_y1_csr(answer, spec)
    if y_id == "Y2":
        return check_y2_csr(answer, spec)
    if y_id == "Y4":
        return check_y4_csr(answer, spec)
    return None


async def judge_dcr(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    q_data: dict,
    answer: str,
    main_version: dict,
) -> DcrJudgeScores:
    rubric = (
        main_version.get("evaluation_rubric_main_set")
        or q_data.get("evaluation_rubric")
        or {}
    )
    user_msg = (
        f"[Original Question]\n{q_data.get('question_text', '')}\n\n"
        f"[Original Ground Truth]\n{q_data.get('ground_truth_answer', '')}\n\n"
        f"[Original Causal Rationale]\n{q_data.get('causal_rationale') or q_data.get('rationale', '')}\n\n"
        f"[Constraint ID]\n{main_version.get('constraint_id', '')}\n\n"
        f"[Constraint Spec]\n{json.dumps(main_version.get('constraint_spec', {}), ensure_ascii=False, indent=2)}\n\n"
        f"[Constrained Question]\n{main_version.get('constrained_question', '')}\n\n"
        f"[Adapted Ground Truth]\n{main_version.get('adapted_ground_truth', '')}\n\n"
        f"[Constrained Evaluation Rubric]\n"
        f"L1 (Perception/Grounding): {rubric.get('L1_perception_check', '')}\n"
        f"L2 (Interaction/Mechanism): {rubric.get('L2_interaction_check', '')}\n"
        f"L3 (Causal Resolution): {rubric.get('L3_causal_check', '')}\n\n"
        f"[VLM Answer]\n{answer}"
    )
    async with sem:
        response = await client.beta.chat.completions.parse(
            model=JUDGE_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": DCR_JUDGE_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format=DcrJudgeScores,
        )
    return response.choices[0].message.parsed


async def judge_one(client: AsyncOpenAI, sem: asyncio.Semaphore, task: dict) -> tuple[dict, bool]:
    q_data = task["q_data"]
    main_version = task["main_version"]
    answer = main_version.get("vlm_answer", "")
    constraint_id = main_version.get("constraint_id", "")
    spec = main_version.get("constraint_spec", {})
    rubric_source = (
        "evaluation_rubric_main_set"
        if main_version.get("evaluation_rubric_main_set")
        else "evaluation_rubric_fallback"
        if q_data.get("evaluation_rubric")
        else "context_only_fallback"
    )

    if not answer or answer.startswith("[Error]"):
        return task, False

    try:
        dcr_scores = await judge_dcr(client, sem, q_data, answer, main_version)
        raw, cascaded, dcr = compute_dcr(
            dcr_scores.r1_score, dcr_scores.r2_score, dcr_scores.r3_score
        )

        csr_result = check_rule_based_csr(answer, constraint_id, spec)
        if csr_result is None:
            csr_score, csr_reasoning = await judge_y3_csr(
                client,
                sem,
                answer,
                main_version.get("constrained_question", ""),
                spec,
            )
        else:
            csr_score, csr_reasoning = csr_result

        main_version["main_set_evaluation_results"] = {
            "raw_scores": raw,
            "cascaded_scores": cascaded,
            "DCR": dcr,
            "CSR": csr_score,
            "effective_DCR": dcr if csr_score == 1 else 0.0,
            "judge_reasoning": dcr_scores.reasoning,
            "constraint_reasoning": csr_reasoning,
            "rubric_source": rubric_source,
        }
        return task, True
    except Exception as exc:
        print(f"\n[Error] Judging failed: {exc}")
        return task, False


async def main() -> None:
    load_project_dotenv()

    print("=" * 70)
    print(f"  Main-Set Judge (Top-100)  |  model: {MODEL_TO_EVALUATE}  judge: {JUDGE_MODEL}")
    print("=" * 70)

    if not INPUT_JSON.exists():
        print(f"[Fatal] {INPUT_JSON.name} not found. Run step2_vlm_inference_main_set.py first.")
        return

    with open(INPUT_JSON, encoding="utf-8") as f:
        dataset: list[dict] = json.load(f)

    rubric_lookup = build_main_set_rubric_lookup(RUBRIC_SOURCE_JSON)
    merged_rubrics = merge_missing_main_set_rubrics(dataset, rubric_lookup)
    if merged_rubrics:
        print(f"  Merged {merged_rubrics} constrained rubrics from {RUBRIC_SOURCE_JSON.name}.")
    elif not rubric_lookup:
        print(f"  [Warning] {RUBRIC_SOURCE_JSON.name} not found.")

    tasks = []
    skipped = 0
    for img in dataset:
        for dim, q_list in img.get("questions", {}).items():
            for q_idx, q_data in enumerate(q_list):
                for var_idx, main_version in enumerate(q_data.get("main_set_versions") or []):
                    if "main_set_evaluation_results" in main_version:
                        skipped += 1
                        continue
                    if main_version.get("vlm_answer"):
                        tasks.append({
                            "q_data": q_data,
                            "main_version": main_version,
                            "dim": dim,
                            "q_idx": q_idx,
                            "var_idx": var_idx,
                        })

    total_q = sum(
        1
        for img in dataset
        for q_list in img.get("questions", {}).values()
        for q in q_list
        for m in (q.get("main_set_versions") or [])
        if m.get("vlm_answer")
    )

    print(f"\n[1/3] {total_q} answered constrained variants")
    if skipped:
        print(f"      Resuming: {skipped} already judged, {len(tasks)} remaining.")

    if tasks:
        print(f"\n[2/3] Judging DCR + CSR [concurrency={MAX_CONCURRENT}] ...")
        client = AsyncOpenAI()
        sem = asyncio.Semaphore(MAX_CONCURRENT)
        results = await tqdm.gather(
            *(judge_one(client, sem, task) for task in tasks),
            desc="  judging",
            unit="q",
        )
        ok = sum(1 for _, success in results if success)
        err = sum(1 for _, success in results if not success)
        print(f"\n      Success: {ok} | Failed: {err}")
    else:
        print("\n[2/3] All answered variants already judged.")

    evaluations = [
        m["main_set_evaluation_results"]
        for img in dataset
        for q_list in img.get("questions", {}).values()
        for q in q_list
        for m in (q.get("main_set_versions") or [])
        if "main_set_evaluation_results" in m
    ]
    print("\n[3/3] Summary")
    if evaluations:
        avg_dcr = sum(r["DCR"] for r in evaluations) / len(evaluations)
        avg_csr = sum(r["CSR"] for r in evaluations) / len(evaluations)
        avg_eff = sum(r["effective_DCR"] for r in evaluations) / len(evaluations)
        print(f"      Scored: {len(evaluations)}")
        print(f"      DCR: {avg_dcr:.3f} | CSR: {avg_csr:.1%} | effective_DCR: {avg_eff:.3f}")
    else:
        print("      No evaluation results found.")

    ensure_parent(OUTPUT_JSON)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {OUTPUT_JSON.name}")


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
