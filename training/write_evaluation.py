"""
Write docs/EVALUATION.md from the saved results (docs/MDRIS_CONTEXT.md sections 14.1, 15)
==========================================================================================
Reads models/vision/metrics.json, models/nlp/metrics.json and models/fusion_analysis.json
and renders tables plus figure links. Nothing is typed in by hand, so the document
cannot drift from the JSON files.

Usage:  python training/write_evaluation.py
"""

import json

from common import MODELS_DIR, REPO_ROOT


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def f(v, d=3):
    return "n/a" if v is None else f"{v:.{d}f}"


def vision_section(m) -> list[str]:
    if m is None:
        return ["## 1. Vision (YOLOv8n)", "", "**Not available:** `models/vision/metrics.json` not found.", ""]
    t = m["metrics"]["test"]
    hp = m["hyperparameters"]
    out = [
        "## 1. Vision: fine-tuned YOLOv8n",
        "",
        f"- Dataset: {m['dataset']['name']} ({m['dataset']['url']}), licence {m['dataset']['licence']}; "
        f"splits {m['dataset']['splits']}",
        f"- Base model: {m['base_model']}; trained {m['trained_at']} on {m['hardware']} in "
        f"{m['train_time_minutes']} min; epochs run {hp['epochs_run']} of {hp['epochs']} (patience {hp['patience']}), "
        f"batch {hp['batch']}, imgsz {hp['imgsz']}, seed {hp['seed']}",
        "",
        "### Test split",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| mAP50 | **{f(t['mAP50'])}** |",
        f"| mAP50-95 | {f(t['mAP50_95'])} |",
        "",
        "| Class | Test instances | Precision | Recall | AP50 |",
        "|---|---|---|---|---|",
    ]
    for cls, c in t["per_class"].items():
        out.append(f"| {cls} | {c.get('instances', 'n/a')} | {f(c['precision'])} | {f(c['recall'])} | {f(c['AP50'])} |")
    base = m.get("baselines", {})
    if base:
        out += ["", "### Baseline: COCO-pretrained YOLOv8n (no fine-tuning)", "",
                f"Method: {base['coco_yolov8n']['method']}.", "",
                "| Class | COCO yolov8n AP50 | Fine-tuned AP50 (same method) | Instances |", "|---|---|---|---|"]
        for cls, c in base["coco_yolov8n"]["per_class"].items():
            ft = base["finetuned_same_method"]["per_class"].get(cls, {})
            out.append(f"| {cls} | {f(c['AP50'])} | {f(ft.get('AP50'))} | {c['instances']} |")
    out += ["", "### Figures", ""]
    for fig in m["figures"]:
        out.append(f"- [{fig}](../models/vision/{fig})")
    if m.get("failure_cases"):
        out += ["", "### Worst 6 test images (FN + FP at conf 0.25, IoU 0.5)", "",
                "| Image | FN | FP | Figure |", "|---|---|---|---|"]
        for fc in m["failure_cases"]:
            out.append(f"| {fc['image']} | {fc['false_negatives']} | {fc['false_positives']} | "
                       f"[{fc['file']}](../models/vision/{fc['file']}) |")
    if m.get("notes"):
        out += ["", "### Notes", ""] + [f"- {n}" for n in m["notes"]]
    return out + [""]


def nlp_section(m) -> list[str]:
    if m is None:
        return ["## 2. Urgency NLP (DistilBERT)", "", "**Not available:** `models/nlp/metrics.json` not found.", ""]
    t = m["metrics"]["test"]
    hp = m["hyperparameters"]
    out = [
        "## 2. Urgency NLP: fine-tuned DistilBERT",
        "",
        f"- Dataset: {m['dataset']['name']} ({m['dataset']['url']}), licence {m['dataset']['licence']}; events "
        f"{', '.join(m['dataset']['events'])}; splits {m['dataset']['splits']}",
        f"- Base model: {m['base_model']}; trained {m['trained_at']} on {m['hardware']} in "
        f"{m['train_time_minutes']} min; epochs run {hp['epochs_run']} of {hp['epochs']}; batch {hp['batch']}, "
        f"lr {hp['learning_rate']}, max_length {hp['max_length']}; loss: {hp['loss']} {hp['class_weights']}",
        f"- Best validation macro-F1: {f(m['validation']['best_val_macro_f1'])}",
        "",
        "### Test split: fine-tuned model vs baselines",
        "",
        "| Model | Accuracy | Macro-F1 | F1 Low | F1 Medium | F1 High |",
        "|---|---|---|---|---|---|",
    ]
    for b in m["baselines"].values():
        name = b["name"] + (f" (`{b['model']}`)" if b.get("model") else "")
        pc = b["per_class_f1"]
        out.append(f"| {name} | {f(b['accuracy'])} | {f(b['macro_f1'])} | {f(pc['Low'])} | {f(pc['Medium'])} | {f(pc['High'])} |")
    pc = t["per_class"]
    out.append(f"| **Fine-tuned DistilBERT** | **{f(t['accuracy'])}** | **{f(t['macro_f1'])}** | {f(pc['Low']['f1'])} | "
               f"{f(pc['Medium']['f1'])} | {f(pc['High']['f1'])} |")
    out += ["", f"Weighted-F1 (fine-tuned): {f(t['weighted_f1'])}", "",
            "| Urgency | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for cls, c in pc.items():
        out.append(f"| {cls} | {f(c['precision'])} | {f(c['recall'])} | {f(c['f1'])} | {c['support']} |")
    labels = t.get("confusion_matrix_labels", ["Low", "Medium", "High"])
    out += ["", "Confusion matrix (rows = true, columns = predicted):", "",
            "| true \\ pred | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for lab, row in zip(labels, t["confusion_matrix"]):
        out.append(f"| {lab} | " + " | ".join(str(x) for x in row) + " |")
    out += ["", "| Event | Test rows | Macro-F1 |", "|---|---|---|"]
    for ev, v in t["per_event_macro_f1"].items():
        out.append(f"| {ev} | {t.get('per_event_n', {}).get(ev, 'n/a')} | {f(v)} |")
    out += ["", "### Figures", ""] + [f"- [{fig}](../models/nlp/{fig})" for fig in m["figures"]]
    if m.get("notes"):
        out += ["", "### Notes", ""] + [f"- {n}" for n in m["notes"]]
    return out + [""]


def fusion_section(a) -> list[str]:
    out = ["## 3. Fusion", "", "### Unit tests T1–T8", "",
           "`backend/tests/test_fusion.py` asserts the exact expected values of section 8.4 (±0.5 on scores, "
           "±0.01 on evidence) plus a test that the fusion function has no weather parameter. "
           "Run: `cd backend && pytest -q`.", ""]
    if a is None:
        return out + ["**Ablation / sensitivity not available:** `models/fusion_analysis.json` not found.", ""]
    out += [f"### Ablation over the {a['n_incidents']} simulated incidents", "",
            f"Default band counts: {a['default']['band_counts']}", "",
            "| Variant | Description | Band changes | Kendall τ (all) | Kendall τ (top-10) | Mean score shift |",
            "|---|---|---|---|---|---|"]
    for r in a["ablation"]:
        out.append(f"| {r['variant']} | {r['description']} | {r['band_changes']} | {f(r['kendall_tau'])} | "
                   f"{f(r['kendall_tau_top10'])} | {r['mean_score_shift']:+.2f} |")
    out += ["", "### Sensitivity", "", "| Setting | Band changes | Kendall τ (all) | Kendall τ (top-10) |", "|---|---|---|---|"]
    for r in a["sensitivity"]["vision_weight"] + a["sensitivity"]["reliability_scale"]:
        out.append(f"| {r['setting']} | {r['band_changes']} | {f(r['kendall_tau_all'])} | {f(r['kendall_tau_top10'])} |")
    if a.get("notes"):
        out += ["", "### Notes", ""] + [f"- {n}" for n in a["notes"]]
    return out + [""]


def main() -> None:
    vision = load(MODELS_DIR / "vision" / "metrics.json")
    nlp = load(MODELS_DIR / "nlp" / "metrics.json")
    fusion = load(MODELS_DIR / "fusion_analysis.json")
    lines = [
        "# MDRIS evaluation",
        "",
        "_Generated by `training/write_evaluation.py` from `models/vision/metrics.json`, `models/nlp/metrics.json` "
        "and `models/fusion_analysis.json`. Do not edit by hand; re-run the script._",
        "",
        "All model numbers are on held-out **test** splits (see `DATA.md`). Fusion has no ground truth, so it is "
        "evaluated by fixed test cases, an ablation and a sensitivity analysis (section 15.3).",
        "",
    ]
    lines += vision_section(vision) + nlp_section(nlp) + fusion_section(fusion)
    out = REPO_ROOT / "docs" / "EVALUATION.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
