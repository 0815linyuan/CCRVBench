from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


PACKAGE_DIR = Path(__file__).resolve().parent
ICCR_ROOT = PACKAGE_DIR.parent
REPO_ROOT = ICCR_ROOT.parent
ICCR_TEST_DIR = REPO_ROOT / "ICCR_test"


def load_project_dotenv() -> None:
    """Load .env from ICCR root then package paths; fix API base URL aliases."""
    from dotenv import load_dotenv

    for env_path in (
        ICCR_ROOT / ".env",
        PACKAGE_DIR / ".env",
        ICCR_TEST_DIR / ".env",
    ):
        if env_path.is_file():
            load_dotenv(env_path)

    alias = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_COMPAT_BASE_URL")
    if alias and not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = alias.strip().strip("\"'")


def _pick_str(*keys: str, default: str) -> str:
    for k in keys:
        v = os.environ.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return default


def _pick_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


load_project_dotenv()


DATA_TOP100_DIR = ICCR_ROOT / "iccr_dataset_top100"

IMAGE_DIR = ICCR_ROOT / "VG_images_test"
LEGACY_IMAGE_DIR = ICCR_TEST_DIR / "VG_images_test"

BENCHMARK_WITH_RUBRICS_TOP100 = PACKAGE_DIR / "benchmark_with_rubrics_top100.json"

SCENE_GRAPHS_TOP100 = DATA_TOP100_DIR / "test_scene_graphs_top100.json"

BENCHMARK_QUESTIONS_TOP100 = PACKAGE_DIR / "benchmark_questions_top100.json"
BENCHMARK_MAIN_SET_TOP100 = PACKAGE_DIR / "benchmark_main_set_top100.json"
BENCHMARK_MAIN_SET_WITH_RUBRICS_TOP100 = (
    PACKAGE_DIR / "benchmark_main_set_with_rubrics_top100.json"
)

MODEL_OUTPUTS_DIR = PACKAGE_DIR / "model_outputs"
ANALYSIS_DIR = PACKAGE_DIR / "analysis_results"

# Defaults match local Qwen2.5-VL-7B when served under this id; override via .env.
MODEL_TO_EVALUATE = _pick_str("MODEL_TO_EVALUATE", "ICCR_VLM_MODEL", default="gemini-3.5-flash")

JUDGE_MODEL = _pick_str("JUDGE_MODEL", "ICCR_JUDGE_MODEL", default="gpt-5.1")


MAX_CONCURRENT_INFERENCE = _pick_int("MAX_CONCURRENT_INFERENCE", default=2)
MAX_CONCURRENT_JUDGE = _pick_int("MAX_CONCURRENT_JUDGE", default=10)
TEMPERATURE = 0.0
MAX_TOKENS = 512
MCI_THRESHOLD = 0.1
VISION_DETAIL = None

DIMENSIONS = ["X1_discovery", "X2_prediction", "X3_diagnosis", "X4_intervention"]


def model_slug(model_name: str = MODEL_TO_EVALUATE) -> str:
    return model_name.replace("/", "_")


MODEL_SLUG = model_slug()

VLM_ANSWERS_MAIN_TOP100 = MODEL_OUTPUTS_DIR / f"vlm_answers_{MODEL_SLUG}_main_set_top100.json"
RESULTS_MAIN_TOP100 = MODEL_OUTPUTS_DIR / f"results_{MODEL_SLUG}_main_set_top100.json"

VLM_ANSWERS_BASELINE_TOP100 = MODEL_OUTPUTS_DIR / f"vlm_answers_{MODEL_SLUG}_top100_baseline.json"


def baseline_results_path() -> Path:
    """Unconstrained judge output for the same Top100 benchmark (for CDI in step4)."""
    return MODEL_OUTPUTS_DIR / f"results_{MODEL_SLUG}_top100_baseline.json"


def existing_path(preferred: Path, legacy: Optional[Path] = None) -> Path:
    if preferred.exists() or legacy is None:
        return preferred
    if legacy.exists():
        return legacy
    return preferred


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_benchmark_image(img_entry: dict) -> Path:
    """
    Resolve JPEG for benchmark rows. Prefers image_path_relative under REPO_ROOT,
    then falls back to ICCR/VG_images_test/{image_id}.jpg.
    """
    iid = img_entry["image_id"]
    rel = img_entry.get("image_path_relative")
    if rel:
        p = (REPO_ROOT / rel).resolve()
        if p.is_file():
            return p
    return (IMAGE_DIR / f"{iid}.jpg").resolve()
