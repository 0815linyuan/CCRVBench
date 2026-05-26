import json
from pathlib import Path
import pandas as pd

def get_average_dcr(json_path: Path) -> float:
    if not json_path.exists():
        return None
        
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
        
    dcrs = []
    for img in data:
        for dim, q_list in img.get("questions", {}).items():
            for q in q_list:
                er = q.get("evaluation_results")
                if er and "DCR" in er:
                    dcrs.append(er["DCR"])
                    
    if not dcrs:
        return None
    return sum(dcrs) / len(dcrs)

def get_y3_effective_dcr(main_set_path: Path) -> float:
    if not main_set_path.exists():
        return None
        
    with open(main_set_path, encoding="utf-8") as f:
        data = json.load(f)
        
    y3_dcrs = []
    for img in data:
        for dim, q_list in img.get("questions", {}).items():
            for q in q_list:
                for v in q.get("main_set_versions", []):
                    if v.get("constraint_id", "").startswith("Y3"):
                        er = v.get("main_set_evaluation_results")
                        if er and "effective_DCR" in er:
                            # Note: effective_DCR implies the constraint was satisfied (CSR=1)
                            # We can also just take effective_DCR as is.
                            y3_dcrs.append(er["effective_DCR"])
                            
    if not y3_dcrs:
        return None
    return sum(y3_dcrs) / len(y3_dcrs)

def main():
    model_slug = "claude-opus-4-7"
    out_dir = Path("model_outputs")
    
    baseline_path = out_dir / f"results_{model_slug}_top100_baseline.json"
    cot_a_path = out_dir / f"results_{model_slug}_top100_baseline_cot_A.json"
    cot_b_path = out_dir / f"results_{model_slug}_top100_baseline_cot_B.json"
    main_set_path = out_dir / f"results_{model_slug}_main_set_top100.json"
    
    results = {
        "1. Vanilla Baseline": get_average_dcr(baseline_path),
        "2. CoT Variant A (Standard)": get_average_dcr(cot_a_path),
        "3. CoT Variant B (Contrastive)": get_average_dcr(cot_b_path),
        "4. Y3 Constrained (effective_DCR)": get_y3_effective_dcr(main_set_path),
    }
    
    print("\n" + "="*50)
    print(" CoT Ablation Study Results (Average DCR)")
    print("="*50)
    
    for label, score in results.items():
        if score is None:
            print(f"{label:35s} : [Not Found/Not Scored Yet]")
        else:
            print(f"{label:35s} : {score:.3f}")
            
    print("="*50)

if __name__ == "__main__":
    main()
