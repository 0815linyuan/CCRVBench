from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm_asyncio
from config import (
    BENCHMARK_MAIN_SET_TOP100,
    BENCHMARK_QUESTIONS_TOP100,
    SCENE_GRAPHS_TOP100,
    load_project_dotenv,
)
from y4_word_normalize import normalize_y4_words

load_project_dotenv()
client = AsyncOpenAI()

MAX_CONCURRENT = 10
MAX_MUTATION_ATTEMPTS = 2
ALLOWED_CONSTRAINTS_BY_DIM = {
    "X1_discovery": ["Y1", "Y2", "Y3", "Y4"],
    "X2_prediction": ["Y1", "Y2", "Y3", "Y4"],
    "X3_diagnosis": ["Y1", "Y2", "Y3", "Y4"],
    "X4_intervention": ["Y1", "Y2", "Y3", "Y4"],
}
DIMENSIONS = ["X1_discovery", "X2_prediction", "X3_diagnosis", "X4_intervention"]

HUMAN_NAMES = {"man", "woman", "person", "child", "kid", "boy", "girl", "player", "batter"}
HUMAN_FORBIDDEN_TERMS = ["man", "woman", "person", "human", "child", "kid", "boy", "girl", "player", "batter"]

Y2_OUTPUT_SUFFIX_ANCHOR = "Coordinate:"
Y2_OUTPUT_SUFFIX = (
    "\n\nYou MUST answer in exactly this two-line structure (English labels, each on its own line):\n"
    "Coordinate: [x, y]\n"
    "Explanation: <one or more complete sentences of causal reasoning; ground the coordinate in the mechanism>\n"
    "Use image pixel coordinates for [x, y]. Do not output a bounding box. No extra lines before Coordinate."
)
Y2_GT_MIN_EXPLANATION_CHARS = 20
Y2_GT_BBOX_MARGIN_FRAC = 0.075  # 7.5% per side (~15% linear tolerance); aligns with step3 CSR


class MainSetMutation(BaseModel):
    constrained_question: str = Field(
        description="The rewritten question incorporating the assigned main-set constraint."
    )
    adapted_ground_truth: str = Field(
        description="Expected answer for the constrained question. For Y2 use two lines: "
        "'Coordinate: [x, y]' and 'Explanation: ...' (see Y2 rule). For Y3 pairs with causal refutation + truth.",
    )
    false_premise: str | None = Field(
        default=None,
        description="For Y3 only, the explicit false premise injected into the question.",
    )


BASE_SYSTEM_PROMPT = """
You are an expert in visual causal adversarial testing. Your task is to rewrite a base causal question into ONE constrained question.

You will receive:
- X dimension
- Base question
- Original ground truth
- A structured constraint_spec prepared by code
- A compact scene graph context

You must follow the exact rule for the assigned Target Constraint below.
Never alter the core physical truth of the base question.
Only alter the phrasing, constraints, and conditions of the inquiry.
Do not invent coordinates; use only the values in constraint_spec.

You are generating a mutation record, not answering the visual question.
Your top-level response must always match the MainSetMutation schema:
- constrained_question: the rewritten question shown to the future VLM.
- adapted_ground_truth: the correct expected answer for that rewritten question.
- false_premise: only for Y3; otherwise null.
Never return plain text, Markdown, explanations, or the future VLM's answer as the
top-level response.
"""
CONSTRAINT_RULES = {
    "Y1": """
[Y1 Entity Symbolization & Distractor Injection]
You MUST eliminate all real object nouns. Refer to entities exclusively by their assigned Node placeholders (e.g., Node_A, Node_B) from the symbol_map.
CRITICAL QUESTION REWRITE RULES:
1. The symbol_map contains the true causal entities PLUS distractor entities. 
2. In your question, you MUST list ALL nodes from the symbol_map with their bounding boxes to establish the visual context. (e.g., "In the scene, there are Node_A [bbox], Node_B [bbox], and Node_C [bbox].")
3. NEVER reveal the answer in the question. Ask the respondent to identify WHICH specific node is causing the action or interacting with the target node, and WHAT the exact physical action is. 
4. Append this strict instruction: "Answer using only the Node placeholders and state the precise physical action. Do NOT use real object names."
*Adapt GT*: The adapted ground truth must identify the correct nodes from the symbol_map and state the correct causal action.
""",

    "Y2": """
[Y2 Spatial Grounding + Causal Masking (Grounded Causal Explanation)]
Rephrase the base question into spatial localization WITHOUT leaking the causal subject.

CAUSAL MASKING (anti leak):
- The constrained_question MUST NOT name, describe, or uniquely identify the causal entity with real category words
  (e.g. no "man", "woman", "car", "dog", shirt colors, relative phrases like "the person pushing the stroller").
- Ask about an UNKNOWN entity or region only: e.g. which single [x,y] in the image marks the source of the effect,
  or which location is responsible for the observed outcome — so the solver must infer the entity from the image.
- You may refer to the EFFECT / PATIENT (what moves, changes, or is acted on) using generic wording already in the base
  question if needed for context, but do NOT point to the-agent-under-test by any identifiable noun phrase.

SPATIAL ASK (content):
- X1 / X3: ask for the pixel [x, y] of the (masked) causal source.
- X2: ask where in the image the main impact/displacement will manifest (single point).
- X4: ask for the pixel [x, y] of the object to remove/intervene on (masked wording).

RESPONDENT OUTPUT (appended by pipeline to the question):
- They must reply with two labeled lines: Coordinate + Explanation (see suffix). Do NOT repeat that contract inside
  constrained_question beyond what the suffix already states.

ADAPTED GROUND TRUTH (you MUST provide; NOT overwritten by code):
- Line 1 exactly of the form: Coordinate: [x, y] using TWO numbers in image pixel space, bracketed as [x, y].
  The point MUST lie on or inside the causal target object's box implied by constraint_spec (target_bbox_xyxy);
  prefer a visually meaningful contact/interaction point if consistent with the scene, not an arbitrary exterior point.
- Line 2 exactly starting Explanation: followed by a short correct causal chain (why that location is the source /
  locus of the mechanism). This text MAY use real object names for grading clarity — it is not shown to the solver as
  the question text.

Never mention "constraint_spec", "target_object_id", or JSON in constrained_question.
""",

    "Y3": """
[Y3 Contradiction Block]
Inject a clearly false causal premise that contradicts the original ground truth or visible causal relation.

The constrained_question MUST include the following four sentences VERBATIM (exact wording, English), in this order,
either at the start or immediately before the false premise, so the respondent cannot miss them:

This is not a counterfactual task.
The textual premise may be false.
Verify it against the image.
If it contradicts the image, first reject the premise.

Additional requirements for the rest of the constrained_question:
1. Present the false premise as a textual CLAIM to check, not as a hypothetical world to accept.
2. Avoid framing that invites counterfactual continuation (e.g. do not ONLY say "Assume that ..." without the
   verbatim block above). If you use "assume" or similar, it must be clear it is a claim to verify, not a world to reason in.
3. Ask the respondent to verify the premise against the image and, if it contradicts the visual evidence,
   to first state the contradiction and reject the premise, then answer using only visual evidence.
4. Then ask the original causal task in a way that requires the correct visual causal reasoning after rejection.

The adapted ground truth must first refute the false premise, then state the correct visual causal reasoning.
Return the false premise in the false_premise field as usual.
""",

    "Y4": """
[Y4 Extreme Compression Bottleneck]
Rewrite the base causal question to ask for the bare causal structure (e.g., Subject Action Target).

CRITICAL QUESTION REWRITE RULES:
1. Instruct the respondent to output ONLY the core causal entities and the action (e.g., person pushes glass or car hits tree).
2. Instruct them to NOT use punctuation except optionally ASCII or Unicode hyphens inside compound tokens (e.g., coffee-cup); no articles (a, an, the) or explanatory text.
3. DO NOT specify any exact word limit in your question text. The pipeline will automatically append the dynamic word limit constraint later.

The adapted ground truth MUST be a maximally concise phrase reflecting this rule (2 to 6 words after normalization: hyphens become spaces, then other punctuation is stripped; the pipeline applies this).
""",
}


def build_graph_lookup(graphs: list[dict]) -> dict[int, dict]:
    return {g["image_id"]: g for g in graphs}


def append_y2_output_suffix(question: str) -> str:
    text = (question or "").rstrip()
    if Y2_OUTPUT_SUFFIX_ANCHOR in text:
        return text
    return text + Y2_OUTPUT_SUFFIX


def _parse_coordinate_numbers_from_slice(text: str) -> tuple[float, float] | None:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(numbers) != 2:
        return None
    return float(numbers[0]), float(numbers[1])


def _y2_expand_xyxy(
    xyxy: list[float] | tuple[float, ...], margin_fraction: float
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    lo_x, hi_x = (x1, x2) if x1 <= x2 else (x2, x1)
    lo_y, hi_y = (y1, y2) if y1 <= y2 else (y2, y1)
    w = hi_x - lo_x
    h = hi_y - lo_y
    dx, dy = w * margin_fraction, h * margin_fraction
    return lo_x - dx, lo_y - dy, hi_x + dx, hi_y + dy


def _y2_point_in_xyxy(pt: tuple[float, float], xyxy: tuple[float, float, float, float]) -> bool:
    x, y = pt
    x1, y1, x2, y2 = xyxy
    lo_x, hi_x = (x1, x2) if x1 <= x2 else (x2, x1)
    lo_y, hi_y = (y1, y2) if y1 <= y2 else (y2, y1)
    return lo_x <= x <= hi_x and lo_y <= y <= hi_y


def validate_y2_adapted_ground_truth(adapted_gt: str, constraint_spec: dict) -> None:
    """Ensure GPT-produced GT has Coordinate + Explanation lines and a plausible in-bbox point."""
    raw = (adapted_gt or "").strip()
    if not raw:
        raise ValueError("Y2 adapted_ground_truth is empty.")

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    coord_line = next((ln for ln in lines if ln.lower().startswith("coordinate:")), None)
    expl_line = next((ln for ln in lines if ln.lower().startswith("explanation:")), None)
    if not coord_line:
        raise ValueError("Y2 adapted_ground_truth must include a line starting with 'Coordinate:'.")
    if not expl_line:
        raise ValueError("Y2 adapted_ground_truth must include a line starting with 'Explanation:'.")

    _, coord_rest = coord_line.split(":", 1)
    point = _parse_coordinate_numbers_from_slice(coord_rest)
    if point is None:
        raise ValueError(
            "Y2 Coordinate line must contain exactly two numeric values for [x, y]."
        )

    _, expl_rest = expl_line.split(":", 1)
    if len(expl_rest.strip()) < Y2_GT_MIN_EXPLANATION_CHARS:
        raise ValueError(
            f"Y2 Explanation must be at least {Y2_GT_MIN_EXPLANATION_CHARS} characters after the colon."
        )

    bbox = constraint_spec.get("target_bbox_xyxy")
    if bbox and len(bbox) == 4:
        expanded = _y2_expand_xyxy(bbox, constraint_spec.get("csr_bbox_margin_fraction", Y2_GT_BBOX_MARGIN_FRAC))
        if not _y2_point_in_xyxy(point, expanded):
            raise ValueError(
                f"Y2 GT coordinate {point} lies outside expanded target bbox (margin-inclusive)."
            )


def build_node_lookup(scene_graph: dict) -> dict[int, dict]:
    return {
        node["object_id"]: node
        for node in scene_graph.get("nodes", [])
        if "object_id" in node
    }


def bbox_metadata(node: dict | None) -> dict[str, Any] | None:
    if not node or "bbox" not in node:
        return None

    x, y, w, h = node["bbox"]
    cx = x + w / 2
    cy = y + h / 2
    return {
        "object_id": node.get("object_id"),
        "true_name": node.get("name", "unknown"),
        "bbox_xywh": [x, y, w, h],
        "bbox_xyxy": [x, y, x + w, y + h],
        "center_xy": [round(cx, 1), round(cy, 1)],
    }


def related_partner_node(base_question: dict, scene_graph: dict, node_lookup: dict[int, dict]) -> dict | None:
    target_id = base_question.get("target_object_id")
    if target_id is None:
        return None

    non_part_edges = [
        edge for edge in scene_graph.get("edges", [])
        if edge.get("predicate") not in {"has", "of"}
    ]

    for edge in non_part_edges:
        if edge.get("target_id") == target_id and edge.get("source_id") in node_lookup:
            return node_lookup[edge["source_id"]]

    for edge in non_part_edges:
        if edge.get("source_id") == target_id and edge.get("target_id") in node_lookup:
            return node_lookup[edge["target_id"]]

    return None


def coordinate_target_node(
    base_question: dict,
    x_dimension: str,
    scene_graph: dict,
    node_lookup: dict[int, dict],
) -> dict | None:
    target = node_lookup.get(base_question.get("target_object_id"))
    partner = related_partner_node(base_question, scene_graph, node_lookup)

    if x_dimension in {"X1_discovery", "X3_diagnosis"}:
        return partner or target
    return target or partner


def forbidden_terms_for(nodes: list[dict | None]) -> list[str]:
    terms: set[str] = set()
    for node in nodes:
        if not node:
            continue
        name = str(node.get("name", "")).strip().lower()
        if not name:
            continue
        terms.add(name)
        terms.add(name.replace("_", " "))
        if name in HUMAN_NAMES:
            terms.update(HUMAN_FORBIDDEN_TERMS)
        if name.endswith("s"):
            terms.add(name[:-1])
        else:
            terms.add(f"{name}s")
    return sorted(terms)


def compact_scene_graph(scene_graph: dict, max_edges: int = 40) -> dict[str, Any]:
    return {
        "image_id": scene_graph.get("image_id"),
        "image_width": scene_graph.get("image_width"),
        "image_height": scene_graph.get("image_height"),
        "nodes": scene_graph.get("nodes", []),
        "edges": scene_graph.get("edges", [])[:max_edges],
    }


def build_constraint_spec(
    base_question: dict, x_dimension: str, constraint_id: str, scene_graph: dict
) -> dict[str, Any]:
    node_lookup = build_node_lookup(scene_graph)
    target_node = node_lookup.get(base_question.get("target_object_id"))
    partner_node = related_partner_node(base_question, scene_graph, node_lookup)
    coordinate_node = coordinate_target_node(base_question, x_dimension, scene_graph, node_lookup)

    target_meta = bbox_metadata(target_node)
    partner_meta = bbox_metadata(partner_node)
    coordinate_meta = bbox_metadata(coordinate_node)

    if constraint_id == "Y1":
        true_nodes = [n for n in [partner_node, target_node] if n]
        true_ids = {n["object_id"] for n in true_nodes}

        all_nodes = scene_graph.get("nodes", [])
        distractors = [n for n in all_nodes if n.get("object_id") not in true_ids and bbox_metadata(n)]

        selected_distractors = random.sample(distractors, min(2, len(distractors)))

        pool = true_nodes + selected_distractors
        random.shuffle(pool)

        symbol_map = {}
        for i, n in enumerate(pool):
            symbol_map[f"Node_{chr(65 + i)}"] = bbox_metadata(n)

        return {
            "csr_rule": "forbidden_term_scan",
            "symbol_map": symbol_map,
            "forbidden_terms": forbidden_terms_for(pool),
        }

    if constraint_id == "Y2":
        return {
            "csr_rule": "grounded_causal_explanation",
            "target_object_id": coordinate_meta["object_id"] if coordinate_meta else None,
            "target_true_name": coordinate_meta["true_name"] if coordinate_meta else None,
            "target_bbox_xywh": coordinate_meta["bbox_xywh"] if coordinate_meta else None,
            "target_bbox_xyxy": coordinate_meta["bbox_xyxy"] if coordinate_meta else None,
            "target_center_xy": coordinate_meta["center_xy"] if coordinate_meta else None,
            "accepted_region": "inside_bbox_expanded",
            "csr_bbox_margin_fraction": Y2_GT_BBOX_MARGIN_FRAC,
        }

    if constraint_id == "Y3":
        return {
            "csr_rule": "contradiction_refusal",
            "original_ground_truth": base_question.get("ground_truth_answer", ""),
            "expected_behavior": "first_refute_false_premise_then_reason_from_visual_evidence",
        }

    if constraint_id == "Y4":
        return {
            "csr_rule": "dynamic_word_limit",
            "forbid_punctuation": True,
            "forbid_extra_text": True,
        }

    raise ValueError(f"Unsupported constraint_id: {constraint_id}")


async def generate_mutation(
    sem: asyncio.Semaphore,
    base_question: dict,
    x_dimension: str,
    scene_graph: dict,
    constraint_id: str,
    image_id: int | None = None,
    q_idx: int | None = None,
) -> dict | None:
    active_rule = CONSTRAINT_RULES.get(constraint_id, "")
    constraint_spec = build_constraint_spec(base_question, x_dimension, constraint_id, scene_graph)

    dynamic_system_prompt = f"""{BASE_SYSTEM_PROMPT}

Target Constraint Rule ({constraint_id}):
{active_rule}
"""

    user_payload = {
        "x_dimension": x_dimension,
        "base_question": base_question["question_text"],
        "original_ground_truth": base_question["ground_truth_answer"],
        "constraint_spec": constraint_spec,
        "scene_graph_context": compact_scene_graph(scene_graph),
    }
    if constraint_id == "Y2":
        # Reduces retries: models often hallucinate contact pixels outside the SG box.
        user_payload["y2_adapted_gt_coordinate_rule"] = (
            "For adapted_ground_truth, the Coordinate line [x,y] MUST lie inside "
            "constraint_spec.target_bbox_xyxy (inclusive of small pipeline margin). "
            "Do not estimate pixels from the image if they can fall outside that rectangle. "
            "Recommended: set Coordinate to constraint_spec.target_center_xy exactly, "
            "or choose any interior point of that same box; align Explanation with that choice."
        )

    context = (
        f"image_id={image_id} dim={x_dimension} q_idx={q_idx} "
        f"constraint_id={constraint_id}"
    )

    async with sem:
        for attempt in range(1, MAX_MUTATION_ATTEMPTS + 1):
            retry_note = ""
            if attempt > 1:
                retry_note = (
                    "\n\nThe previous response failed validation. "
                    "Return only a valid MainSetMutation object. "
                    "Do not answer the visual question."
                )
                if constraint_id == "Y2":
                    retry_note += (
                        f" For Y2, adapted_ground_truth must contain two lines starting with "
                        f"'Coordinate:' and 'Explanation:'; Explanation must be at least "
                        f"{Y2_GT_MIN_EXPLANATION_CHARS} characters after the colon; "
                        f"Coordinate [x,y] must fall inside the expanded target box in constraint_spec "
                        f"(see target_bbox_xyxy + margin {Y2_GT_BBOX_MARGIN_FRAC}). "
                        f"Causal masking: constrained_question must not identify the causal agent by name/appearance."
                    )
                if constraint_id == "Y4":
                    retry_note += (
                        " For Y4, adapted_ground_truth must yield 2-6 words after the same normalization as scoring "
                        "(hyphen-like characters become spaces, then other punctuation is removed). "
                        "No articles a/an/the in the phrase. Do not put a numeric word limit in constrained_question."
                    )
            try:
                system_prompt = dynamic_system_prompt + retry_note
                response = await client.beta.chat.completions.parse(
                    model="gpt-5.1",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                    ],
                    response_format=MainSetMutation,
                    temperature=0.2 if attempt == 1 else 0.0,
                )
                parsed = response.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("empty parsed response")

                if constraint_id == "Y3" and parsed.false_premise:
                    constraint_spec["false_premise"] = parsed.false_premise

                adapted_gt = parsed.adapted_ground_truth
                constrained_question = parsed.constrained_question
                if constraint_id == "Y2":
                    center = constraint_spec.get("target_center_xy")
                    if (
                        not isinstance(center, list)
                        or len(center) != 2
                        or not all(isinstance(v, (int, float)) for v in center)
                    ):
                        print(
                            f"Mutation generation failed for Y2: missing target_center_xy "
                            f"({context}); center={center!r}"
                        )
                        return None
                    try:
                        validate_y2_adapted_ground_truth(adapted_gt, constraint_spec)
                    except ValueError as ve:
                        raise ValueError(f"Y2 adapted_ground_truth validation: {ve}") from ve
                    constrained_question = append_y2_output_suffix(constrained_question)

                if constraint_id == "Y4":
                    words = normalize_y4_words(adapted_gt)
                    gt_len = len(words)
                    clean_gt = " ".join(words)
                    if gt_len < 2 or gt_len > 6:
                        raise ValueError(
                            f"Y4 violation: Ground truth '{clean_gt}' has {gt_len} words, "
                            f"out of allowed range (2-6)."
                        )
                    allowed_max_words = gt_len + 1
                    constraint_spec["allowed_max_words"] = allowed_max_words
                    y4_suffix = (
                        f"\n\nConstraint: Your answer MUST be {allowed_max_words} words or fewer. "
                        "Do NOT use any punctuation (hyphens in compound words are allowed)."
                    )
                    constrained_question = (constrained_question or "").rstrip() + y4_suffix
                    adapted_gt = clean_gt

                return {
                    "constraint_id": constraint_id,
                    "constraint_name": {
                        "Y1": "entity_symbolization",
                        "Y2": "spatial_grounded_causal_explanation",
                        "Y3": "contradiction_block",
                        "Y4": "strict_grammar_bottleneck",
                    }[constraint_id],
                    "x_dimension": x_dimension,
                    "constrained_question": constrained_question,
                    "adapted_ground_truth": adapted_gt,
                    "constraint_spec": constraint_spec,
                }
            except Exception as exc:
                if attempt < MAX_MUTATION_ATTEMPTS:
                    print(
                        f"Mutation generation retrying ({attempt}/{MAX_MUTATION_ATTEMPTS}) "
                        f"for {context}: {type(exc).__name__}: {exc}"
                    )
                    continue
                print(
                    f"Mutation generation failed after {MAX_MUTATION_ATTEMPTS} attempts "
                    f"for {context}: {type(exc).__name__}: {exc}"
                )
                return None


def reset_main_set_versions(questions_data: list[dict]) -> None:
    for img_data in questions_data:
        for dim in DIMENSIONS:
            for q in img_data.get("questions", {}).get(dim, []):
                q.pop("main_set_version", None)
                q["main_set_versions"] = []


async def main() -> None:
    if not BENCHMARK_QUESTIONS_TOP100.exists() or not SCENE_GRAPHS_TOP100.exists():
        print("Required input files are missing.")
        return

    random.seed(42)

    with open(BENCHMARK_QUESTIONS_TOP100, encoding="utf-8") as f:
        questions_data = json.load(f)

    with open(SCENE_GRAPHS_TOP100, encoding="utf-8") as f:
        graphs_data = json.load(f)

    graph_lookup = build_graph_lookup(graphs_data)
    reset_main_set_versions(questions_data)

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = []
    task_references: list[tuple[dict, str, int, str]] = []

    for img_data in questions_data:
        image_id = img_data["image_id"]
        scene_graph = graph_lookup.get(image_id, {})

        for dim in DIMENSIONS:
            for q_idx, q in enumerate(img_data.get("questions", {}).get(dim, [])):
                for constraint_id in ALLOWED_CONSTRAINTS_BY_DIM[dim]:
                    tasks.append(
                        generate_mutation(
                            sem,
                            q,
                            dim,
                            scene_graph,
                            constraint_id,
                            image_id=image_id,
                            q_idx=q_idx,
                        )
                    )
                    task_references.append((img_data, dim, q_idx, constraint_id))

    print(f"Start generating {len(tasks)} main-set mutations (full X×Y per base question).")
    # results = await asyncio.gather(*tasks)
    results = await tqdm_asyncio.gather(*tasks, desc="step1b mutations")

    success_count = 0
    for ref, result in zip(task_references, results):
        img_data, dim, q_idx, _cid = ref
        q = img_data["questions"][dim][q_idx]
        if result:
            q["main_set_versions"].append(result)
            success_count += 1

    print(f"Successfully generated {success_count} mutations.")

    with open(BENCHMARK_MAIN_SET_TOP100, "w", encoding="utf-8") as f:
        json.dump(questions_data, f, ensure_ascii=False, indent=2)

    print(f"Saved main set questions to {BENCHMARK_MAIN_SET_TOP100.name}.")


if __name__ == "__main__":
    asyncio.run(main())
