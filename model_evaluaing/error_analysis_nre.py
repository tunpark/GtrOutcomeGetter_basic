import os
import re
import json
import warnings
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification

import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Focus labels for detailed analysis
FOCUS_LABELS = ["Support_url", "Software"]

# Data loading functions
def load_jsonl_format(path: str):
    texts, all_entities, bad = [], [], 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                text = obj["text"]
                ents = []
                for e in obj.get("entities", []):
                    s0 = int(e["start_offset"])
                    e0 = int(e["end_offset"])
                    lab = str(e["label"])
                    if 0 <= s0 < e0 <= len(text):
                        ents.append((s0, e0, lab))
                texts.append(text)
                all_entities.append(ents)
            except Exception:
                bad += 1
    return texts, all_entities, bad

def load_bio_format(path: str):
    texts, bios = [], []
    toks, labs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            if not s:
                if toks:
                    texts.append(" ".join(toks))
                    bios.append(labs.copy())
                    toks, labs = [], []
            else:
                parts = s.split()
                if len(parts) >= 2:
                    toks.append(parts[0])
                    labs.append(parts[1])
        if toks:
            texts.append(" ".join(toks)); bios.append(labs.copy())

    all_entities = []
    for text, seq in zip(texts, bios):
        entities = []
        pos = 0
        spans = []
        for tok in text.split():
            i = text.find(tok, pos)
            j = i + len(tok)
            spans.append((i, j))
            pos = j + 1
        cur = None
        for (i, j), tag in zip(spans, seq):
            if tag.startswith("B-"):
                if cur:
                    entities.append((cur["s"], cur["t"], cur["lab"]))
                cur = {"s": i, "t": j, "lab": tag[2:]}
            elif tag.startswith("I-") and cur and tag[2:] == cur["lab"]:
                cur["t"] = j
            else:
                if cur:
                    entities.append((cur["s"], cur["t"], cur["lab"]))
                    cur = None
        if cur:
            entities.append((cur["s"], cur["t"], cur["lab"]))
        all_entities.append(entities)
    return texts, all_entities, 0

def load_ner_data(path: str):
    if path.endswith(".jsonl") or path.endswith(".json"):
        return load_jsonl_format(path)
    else:
        return load_bio_format(path)

# Entity decoding with confidence scores
def decode_entities_from_logits(
    text: str,
    logits: torch.Tensor,
    offsets,
    id2label: Dict[int, str]
) -> List[Dict[str, Any]]:
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    pred_ids = probs.argmax(axis=-1)
    seq = []
    for k, (s, t) in enumerate(offsets):
        if s == 0 and t == 0:
            lab = "O"; p = 1.0
        else:
            lab = id2label.get(int(pred_ids[k]), "O")
            p = float(probs[k, pred_ids[k]])
        seq.append(((s, t), lab, p))

    entities = []
    cur = None
    for (s, t), lab, p in seq:
        if lab.startswith("B-"):
            if cur:
                cur["text"] = text[cur["start"]:cur["end"]]
                cur["confidence"] = float(np.mean(cur["scores"]))
                entities.append(cur)
            cur = {"start": s, "end": t, "label": lab[2:], "scores": [p]}
        elif lab.startswith("I-") and cur and lab[2:] == cur["label"]:
            cur["end"] = t
            cur["scores"].append(p)
        else:
            if cur:
                cur["text"] = text[cur["start"]:cur["end"]]
                cur["confidence"] = float(np.mean(cur["scores"]))
                entities.append(cur)
                cur = None
    if cur:
        cur["text"] = text[cur["start"]:cur["end"]]
        cur["confidence"] = float(np.mean(cur["scores"]))
        entities.append(cur)
    return entities

# Entity alignment and classification
def span_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0

def analyze_pairs(
    text: str,
    true_ents: List[Tuple[int, int, str]],
    pred_ents: List[Dict[str, Any]],
    iou_thr_type=0.5,
    iou_thr_boundary=0.1,
):
    rows = []
    used_pred = set()

    for (ts, te, tl) in true_ents:
        best_j, best_iou = -1, 0.0
        for j, p in enumerate(pred_ents):
            iou = span_iou((ts, te), (p["start"], p["end"]))
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_j == -1 or best_iou < iou_thr_boundary:
            rows.append({
                "category": "missed",
                "true_label": tl, "pred_label": "",
                "true_span": (ts, te), "pred_span": "",
                "overlap": best_iou, "pred_confidence": np.nan,
                "true_text": text[ts:te], "pred_text": "", "full_text": text,
            })
        else:
            used_pred.add(best_j)
            p = pred_ents[best_j]
            if p["label"] == tl and p["start"] == ts and p["end"] == te:
                rows.append({
                    "category": "correct",
                    "true_label": tl, "pred_label": p["label"],
                    "true_span": (ts, te), "pred_span": (p["start"], p["end"]),
                    "overlap": 1.0, "pred_confidence": float(p.get("confidence", np.nan)),
                    "true_text": text[ts:te], "pred_text": p.get("text", text[p["start"]:p["end"]]), "full_text": text,
                })
            elif p["label"] == tl and best_iou >= iou_thr_boundary:
                rows.append({
                    "category": "boundary_error",
                    "true_label": tl, "pred_label": p["label"],
                    "true_span": (ts, te), "pred_span": (p["start"], p["end"]),
                    "overlap": best_iou, "pred_confidence": float(p.get("confidence", np.nan)),
                    "true_text": text[ts:te], "pred_text": p.get("text", text[p["start"]:p["end"]]), "full_text": text,
                })
            elif best_iou >= iou_thr_type:
                rows.append({
                    "category": "class_error",
                    "true_label": tl, "pred_label": p["label"],
                    "true_span": (ts, te), "pred_span": (p["start"], p["end"]),
                    "overlap": best_iou, "pred_confidence": float(p.get("confidence", np.nan)),
                    "true_text": text[ts:te], "pred_text": p.get("text", text[p["start"]:p["end"]]), "full_text": text,
                })
            else:
                rows.append({
                    "category": "missed",
                    "true_label": tl, "pred_label": "",
                    "true_span": (ts, te), "pred_span": "",
                    "overlap": best_iou, "pred_confidence": np.nan,
                    "true_text": text[ts:te], "pred_text": "", "full_text": text,
                })

    for j, p in enumerate(pred_ents):
        if j in used_pred:
            continue
        ious = [span_iou((p["start"], p["end"]), (ts, te)) for (ts, te, _) in true_ents]
        max_iou = max(ious) if ious else 0.0
        if max_iou < iou_thr_boundary:
            rows.append({
                "category": "spurious",
                "true_label": "", "pred_label": p["label"],
                "true_span": "", "pred_span": (p["start"], p["end"]),
                "overlap": max_iou, "pred_confidence": float(p.get("confidence", np.nan)),
                "true_text": "", "pred_text": p.get("text", text[p["start"]:p["end"]]), "full_text": text,
            })

    return rows

# Enhanced plotting functions
def savefig(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")

def plot_confusion_matrix(df: pd.DataFrame, outdir: str, normalize='true'):
    """Plot confusion matrix for class errors"""
    plots = os.path.join(outdir, "plots")
    os.makedirs(plots, exist_ok=True)
    
    # Only include class errors for confusion matrix
    class_errors = df[df["category"] == "class_error"]
    if len(class_errors) == 0:
        print("No class errors found for confusion matrix")
        return
    
    # Get all labels
    all_labels = sorted(set(class_errors["true_label"].unique()) | set(class_errors["pred_label"].unique()))
    
    # Create confusion matrix
    cm = confusion_matrix(class_errors["true_label"], class_errors["pred_label"], labels=all_labels)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                xticklabels=all_labels, yticklabels=all_labels, ax=ax, cmap='Blues')
    ax.set_title(f'Confusion Matrix (Class Errors Only, {normalize})')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    savefig(fig, plots, f"confusion_matrix_{normalize}.png")

def plot_f1_comparison(metrics_df: pd.DataFrame, outdir: str):
    """Plot F1 score comparison across all labels"""
    plots = os.path.join(outdir, "plots")
    
    # Sort by strict F1 score
    sorted_df = metrics_df.sort_values('f1_strict', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_df) * 0.4)))
    y_pos = np.arange(len(sorted_df))
    
    ax.barh(y_pos - 0.2, sorted_df['f1_strict'], 0.4, label='Strict F1', alpha=0.8)
    ax.barh(y_pos + 0.2, sorted_df['f1_relaxed'], 0.4, label='Relaxed F1', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_df['label'])
    ax.set_xlabel('F1 Score')
    ax.set_title('F1 Score Comparison by Entity Type')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (strict, relaxed) in enumerate(zip(sorted_df['f1_strict'], sorted_df['f1_relaxed'])):
        ax.text(strict + 0.01, i - 0.2, f'{strict:.3f}', va='center', fontsize=9)
        ax.text(relaxed + 0.01, i + 0.2, f'{relaxed:.3f}', va='center', fontsize=9)
    
    savefig(fig, plots, "f1_comparison_by_entity_type.png")

def plot_precision_recall_curve(df: pd.DataFrame, outdir: str):
    """Plot precision-recall curves for each entity type"""
    plots = os.path.join(outdir, "plots")
    
    # Get all entity types that have predictions with confidence scores
    pred_with_conf = df[~df["pred_confidence"].isna() & (df["pred_label"] != "")]
    if len(pred_with_conf) == 0:
        print("No predictions with confidence scores found")
        return
    
    entity_types = sorted(pred_with_conf["pred_label"].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, entity_type in enumerate(entity_types[:4]):  # Show top 4 entity types
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Get data for this entity type
        entity_data = pred_with_conf[pred_with_conf["pred_label"] == entity_type].copy()
        
        if len(entity_data) == 0:
            ax.text(0.5, 0.5, f'No data for {entity_type}', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Create binary labels (1 for correct, 0 for incorrect)
        y_true = (entity_data["category"] == "correct").astype(int)
        y_scores = entity_data["pred_confidence"]
        
        if len(np.unique(y_true)) < 2:
            ax.text(0.5, 0.5, f'No variation in {entity_type}', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap_score = average_precision_score(y_true, y_scores)
        
        ax.plot(recall, precision, label=f'AP = {ap_score:.3f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'PR Curve: {entity_type}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Remove unused subplots
    for j in range(len(entity_types), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    savefig(fig, plots, "precision_recall_curves.png")

def plot_error_distribution(df: pd.DataFrame, outdir: str):
    """Plot error type distribution and confidence analysis"""
    plots = os.path.join(outdir, "plots")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Error category distribution
    error_counts = df["category"].value_counts()
    colors = ['green' if cat == 'correct' else 'red' for cat in error_counts.index]
    bars = ax1.bar(error_counts.index, error_counts.values, color=colors, alpha=0.7)
    ax1.set_title('Error Category Distribution')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, error_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + count*0.01, 
                str(count), ha='center', va='bottom')
    
    # 2. Confidence distribution by correctness
    has_pred = df[~df["pred_confidence"].isna()]
    if len(has_pred) > 0:
        correct_conf = has_pred[has_pred["category"] == "correct"]["pred_confidence"]
        incorrect_conf = has_pred[has_pred["category"] != "correct"]["pred_confidence"]
        
        ax2.hist(correct_conf, bins=20, alpha=0.7, label='Correct', density=True, color='green')
        ax2.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', density=True, color='red')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Density')
        ax2.set_title('Confidence Distribution')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    # 3. Entity length vs accuracy
    df_copy = df.copy()
    df_copy['true_length'] = df_copy.apply(
        lambda x: len(x['true_text']) if x['true_text'] else 0, axis=1
    )
    df_copy['length_bin'] = pd.cut(df_copy['true_length'], bins=5, labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
    
    length_accuracy = df_copy.groupby('length_bin', observed=True)['category'].apply(
        lambda x: (x == 'correct').sum() / len(x)
    ).dropna()
    
    if len(length_accuracy) > 0:
        ax3.bar(range(len(length_accuracy)), length_accuracy.values, alpha=0.7)
        ax3.set_xticks(range(len(length_accuracy)))
        ax3.set_xticklabels(length_accuracy.index, rotation=45)
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy by Entity Length')
        ax3.grid(axis='y', alpha=0.3)
    
    # 4. Top error-prone entity types
    error_by_type = df[df["category"] != "correct"].groupby("true_label").size().sort_values(ascending=False).head(8)
    if len(error_by_type) > 0:
        ax4.barh(range(len(error_by_type)), error_by_type.values, alpha=0.7)
        ax4.set_yticks(range(len(error_by_type)))
        ax4.set_yticklabels(error_by_type.index)
        ax4.set_xlabel('Error Count')
        ax4.set_title('Most Error-Prone Entity Types')
        ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    savefig(fig, plots, "error_distribution_analysis.png")

# Metrics computation
def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def compute_label_metrics(df: pd.DataFrame, label: str):
    true_support = int(((df["true_label"] == label) & df["true_label"].notna()).sum())
    pred_support = int(((df["pred_label"] == label) & df["pred_label"].notna()).sum())

    # Strict metrics
    tp_strict = int(((df.category == "correct") & (df.true_label == label)).sum())
    fn_strict = int(((df.true_label == label) & (df.category != "correct")).sum())
    fp_strict = int(((df.pred_label == label) & (df.category.isin(["boundary_error", "class_error", "spurious"]))).sum())
    prec_strict = _safe_div(tp_strict, tp_strict + fp_strict)
    rec_strict = _safe_div(tp_strict, tp_strict + fn_strict)
    f1_strict = _safe_div(2 * prec_strict * rec_strict, prec_strict + rec_strict) if (prec_strict + rec_strict) else 0.0

    # Relaxed metrics
    ok_relaxed = (df.true_label == label) & (df.pred_label == label) & (df.overlap.fillna(0) >= 0.5)
    tp_relaxed = int(ok_relaxed.sum())
    fn_relaxed = int(((df.true_label == label) & (~ok_relaxed)).sum())
    fp_relaxed = int(((df.pred_label == label) & (~ok_relaxed)).sum())
    prec_relaxed = _safe_div(tp_relaxed, tp_relaxed + fp_relaxed)
    rec_relaxed = _safe_div(tp_relaxed, tp_relaxed + fn_relaxed)
    f1_relaxed = _safe_div(2 * prec_relaxed * rec_relaxed, prec_relaxed + rec_relaxed) if (prec_relaxed + rec_relaxed) else 0.0

    return {
        "label": label,
        "true_support": true_support,
        "pred_support": pred_support,
        "tp_strict": tp_strict, "fp_strict": fp_strict, "fn_strict": fn_strict,
        "precision_strict": round(prec_strict, 4), "recall_strict": round(rec_strict, 4), "f1_strict": round(f1_strict, 4),
        "tp_relaxed": tp_relaxed, "fp_relaxed": fp_relaxed, "fn_relaxed": fn_relaxed,
        "precision_relaxed": round(prec_relaxed, 4), "recall_relaxed": round(rec_relaxed, 4), "f1_relaxed": round(f1_relaxed, 4),
    }

def safe_name(s: str):
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)

def export_all_label_metrics(df: pd.DataFrame, outdir: str, sort_by: str = "f1_strict"):
    labels = sorted(set(df["true_label"].dropna().unique()).union(set(df["pred_label"].dropna().unique())))
    rows = [compute_label_metrics(df, lab) for lab in labels]
    metrics_df = pd.DataFrame(rows).sort_values(sort_by, ascending=False)
    path = os.path.join(outdir, "label_metrics_all.csv")
    metrics_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"All label metrics exported: {path}")
    return metrics_df

def export_focus_reports(df: pd.DataFrame, outdir: str, focus_labels: List[str], top_k=50):
    """Export detailed reports for focus labels"""
    os.makedirs(outdir, exist_ok=True)
    metrics_rows = []
    
    for label in focus_labels:
        # Get subset for this label
        sub = df[((df.true_label == label) | (df.pred_label == label) | 
                 ((df.category == "spurious") & (df.pred_label == label)))]
        
        # Compute metrics
        metrics = compute_label_metrics(df, label)
        metrics_rows.append(metrics)
        
        # Save individual label metrics
        mdf = pd.DataFrame([metrics])
        mdf.to_csv(os.path.join(outdir, f"focus_{safe_name(label)}_metrics.csv"), 
                  index=False, encoding="utf-8-sig")
        
        # Export error examples (Top K by confidence)
        err = sub[sub.category != "correct"].copy()
        if len(err) > 0:
            err["_score"] = err["pred_confidence"].fillna(0.0)
            topn = min(top_k, len(err))
            top_err = err.sort_values("_score", ascending=False).head(topn).drop(columns=["_score"])
        else:
            top_err = err
        
        top_csv = os.path.join(outdir, f"focus_{safe_name(label)}_errors_top{len(top_err)}.csv")
        top_err.to_csv(top_csv, index=False, encoding="utf-8-sig")
        
        # Confusion analysis - true=label被错成什么
        ce_true = df[(df.category == "class_error") & (df.true_label == label)]
        conf_true = ce_true.groupby("pred_label").size().reset_index(name="count").sort_values("count", ascending=False)
        conf_true.to_csv(os.path.join(outdir, f"focus_{safe_name(label)}_confusions_true_{safe_name(label)}.csv"), 
                        index=False, encoding="utf-8-sig")
        
        # Confusion analysis - pred=label实际是什么
        ce_pred = df[(df.category == "class_error") & (df.pred_label == label)]
        conf_pred = ce_pred.groupby("true_label").size().reset_index(name="count").sort_values("count", ascending=False)
        conf_pred.to_csv(os.path.join(outdir, f"focus_{safe_name(label)}_confusions_pred_{safe_name(label)}.csv"), 
                        index=False, encoding="utf-8-sig")
    
    # Export summary metrics for all focus labels
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(outdir, "focus_labels_metrics_all.csv"), 
                     index=False, encoding="utf-8-sig")
    print(f"Focus labels analysis exported for: {', '.join(focus_labels)}")
    return metrics_df

# Main analysis function
def run_ner_error_analysis(
    data_file="cleaned_all.jsonl",
    model_dir="transformer_ner_model",
    output_dir="analysis_ner_results",
    max_length=256,
    top_k=50
):
    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if getattr(model.config, "id2label", None):
        id2label = {int(k): v for k, v in model.config.id2label.items()}
    else:
        id2label = {i: l for l, i in model.config.label2id.items()}

    # Load and split data
    texts, entities, skipped = load_ner_data(data_file)
    if skipped:
        print(f"Skipped {skipped} invalid samples")
    if not texts:
        raise ValueError("No valid samples found!")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, entities, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"Data split: train {len(X_train)} | test {len(X_test)}")

    # Run inference on test set
    all_rows = []
    for text, gold_ents in zip(X_test, y_test):
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        offsets = enc["offset_mapping"].squeeze(0).tolist()

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits.squeeze(0)

        pred_ents = decode_entities_from_logits(text, logits, offsets, id2label)
        rows = analyze_pairs(text, gold_ents, pred_ents, iou_thr_type=0.5, iou_thr_boundary=0.1)
        all_rows.extend(rows)

    # Create analysis dataframe
    df = pd.DataFrame(all_rows)
    full_csv = os.path.join(output_dir, "ner_entity_analysis.csv")
    df.to_csv(full_csv, index=False, encoding="utf-8-sig")
    print(f"Full analysis saved: {full_csv}")

    # Export metrics for all labels
    all_label_metrics = export_all_label_metrics(df, output_dir)

    # Generate quantitative plots
    print("Generating quantitative analysis plots...")
    plot_confusion_matrix(df, output_dir, normalize='true')
    plot_f1_comparison(all_label_metrics, output_dir)
    plot_precision_recall_curve(df, output_dir)
    plot_error_distribution(df, output_dir)

    # Print key metrics
    if "support_url" in all_label_metrics["label"].values:
        row = all_label_metrics[all_label_metrics["label"] == "support_url"].iloc[0]
        print(f"support_url F1 - strict: {row['f1_strict']:.4f}, relaxed: {row['f1_relaxed']:.4f}")

    print("NER error analysis with quantitative plots completed.")
    return {
        "full_csv": full_csv,
        "metrics_csv": os.path.join(output_dir, "label_metrics_all.csv"),
        "plots_dir": os.path.join(output_dir, "plots")
    }

if __name__ == "__main__":
    DATA_FILE = "cleaned_all.jsonl"
    MODEL_DIR = "transformer_ner_model"
    OUTPUT_DIR = "analysis_ner_results"

    run_ner_error_analysis(
        data_file=DATA_FILE,
        model_dir=MODEL_DIR,
        output_dir=OUTPUT_DIR,
        max_length=256,
        top_k=50

    )
