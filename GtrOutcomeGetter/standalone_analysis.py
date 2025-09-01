"""
GTR NRE + Intelligent Fallback 
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import traceback
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AnalyzerConfirmedLogic:
    MAIN_OUTCOME_TYPES = {
        "ARTISTIC_AND_CREATIVE_PRODUCT": ["ARTISTIC_AND_CREATIVE_PRODUCT"],
        "KEY_FINDINGS": ["KEY_FINDINGS"],
        "RESEARCH_DATABASES_AND_MODELS": ["RESEARCH_DATABASES_AND_MODELS"],
        "SOFTWARE_AND_TECHNICAL_PRODUCT": ["SOFTWARE_AND_TECHNICAL_PRODUCT"],
        "RESEARCH_TOOLS_AND_METHODS": ["RESEARCH_TOOLS_AND_METHODS"],
        "ENGAGEMENT_ACTIVITY": ["ENGAGEMENT_ACTIVITY"],
        "INTELLECTUAL_PROPERTY": ["INTELLECTUAL_PROPERTY"]
    }

    def __init__(self, db_path: str = "projects_sample.db"):
        # Locate database
        possible_paths = [
            db_path, f"../getData/{db_path}", f"getData/{db_path}",
            "../getData/outcomes.db", "getData/outcomes.db",
            "../getData/projects_sample.db", "getData/projects_sample.db",
            f"../{db_path}", f"../../{db_path}", "outcomes.db"
        ]
        self.db_path = None
        for p in possible_paths:
            if Path(p).exists():
                self.db_path = Path(p); break
        if self.db_path is None:
            raise FileNotFoundError(f"Database file not found, tried paths: {possible_paths}")
        logger.info(f"Connected to database: {self.db_path}")

        self.classifier_model = None
        self.ner_model = None
        self.models_status = {"classifier": {"loaded": False, "error": None},
                              "ner": {"loaded": False, "error": None}}
        self._load_models()

    def _load_models(self):
        import os, warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        warnings.filterwarnings("ignore")
        
        # Load classifier
        try:
            if not Path("classification_loader.py").exists():
                raise FileNotFoundError("classification_loader.py does not exist")
            from classification_loader import load_latest_model
            self.classifier_model = load_latest_model()
            if self.classifier_model:
                self.models_status["classifier"]["loaded"] = True
                logger.info("Classifier loaded successfully")
            else:
                self.models_status["classifier"]["error"] = "load_latest_model returned None"
        except Exception as e:
            self.models_status["classifier"]["error"] = str(e)
            logger.warning(f"Classifier unavailable: {e}")
        
        # Load NER
        try:
            if not Path("ner_model_loader.py").exists():
                raise FileNotFoundError("ner_model_loader.py does not exist")
            from ner_model_loader import load_ner_model
            self.ner_model = load_ner_model()
            if self.ner_model:
                self.models_status["ner"]["loaded"] = True
                logger.info("NER model loaded successfully")
            else:
                self.models_status["ner"]["error"] = "load_ner_model returned None"
        except Exception as e:
            self.models_status["ner"]["error"] = str(e)
            logger.warning(f"NER unavailable: {e}")

    def _map_main_type(self, t: str) -> str:
        if not t:
            return "UNKNOWN"
        for main_type, subs in self.MAIN_OUTCOME_TYPES.items():
            if t in subs:
                return main_type
        return "OTHER"

    def _classifier_says_software(self, text: str) -> (bool, float):
        """Use classifier only to determine if text is software, return (is_software, confidence)"""
        if not self.classifier_model:
            return (False, 0.0)
        try:
            cls = self.classifier_model.predict(text)
            pred = str(cls.get("predicted_class", "")).lower()
            conf = float(cls.get("confidence", 0.0))
            is_soft = (pred == "1") or (pred == "software") or ("software" in pred)
            return (is_soft, conf)
        except Exception:
            return (False, 0.0)

    def _ner_extract_count(self, description: str) -> int:
        if not description or not self.ner_model:
            return 0
        try:
            ents = self.ner_model.predict(description) or []
            ents = [e for e in ents if str(e.get("label","")).lower() != "software_url"]
            sw = [e["text"] for e in ents if str(e.get("label","")).lower() == "software"]
            return len(list(dict.fromkeys(sw)))
        except Exception:
            return 0

    def run(self, limit: int = 80000) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path); conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        try:
            cur.execute("SELECT * FROM outcomes LIMIT ?", (limit,))
            rows = [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

        if not rows:
            return {"error": "outcomes table empty or no data retrieved"}

        # Statistics table
        table = {mt: {"Total": 0, "Classifier Software Count": 0, "TRUE Count": 0}
                 for mt in list(self.MAIN_OUTCOME_TYPES.keys()) + ["OTHER","UNKNOWN"]}

        # TRUE list (for GT alignment)
        true_predictions = []  # Only collect TRUE outcome_ids

        for i, row in enumerate(rows):
            outcome_id = row.get('outcome_id', f'outcome_{i+1}')
            title = (row.get('title') or '').strip()
            description = (row.get('description') or '').strip()
            t = row.get('type') or row.get('outcome_type', 'UNKNOWN')
            main_type = self._map_main_type(t)

            table[main_type]["Total"] += 1

            text = (title + " " + description).strip()
            if not text:
                continue

            # 1) Classifier judgment
            is_soft_by_clf, conf = self._classifier_says_software(text)
            if is_soft_by_clf:
                table[main_type]["Classifier Software Count"] += 1

                # 2) Enter NER
                ner_cnt = self._ner_extract_count(description)

                # 3) TRUE determination
                is_true = False
                if ner_cnt > 0:
                    is_true = True
                elif title:
                    # NER has no entities -> title intelligent fallback
                    is_true = True

                if is_true:
                    table[main_type]["TRUE Count"] += 1
                    true_predictions.append({"outcome_id": outcome_id, "type": t})

            # Classifier judges non-software: don't enter NER, no fallback, not marked as TRUE

            if (i+1) % 5000 == 0:
                logger.info(f"Processed {i+1:,}/{len(rows):,}")

        return {
            "table": table,
            "true_predictions": true_predictions,  # Only contains TRUE outcome_id/type
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "db_file": str(self.db_path),
            "models_status": self.models_status
        }

    def load_gt(self, excel_file_path: str = "software.xlsx") -> Optional[pd.DataFrame]:
        paths = [excel_file_path, f"getData/{excel_file_path}", f"../getData/{excel_file_path}", f"../{excel_file_path}"]
        for p in paths:
            if Path(p).exists():
                try:
                    df = pd.read_excel(p)
                    return df
                except Exception as e:
                    logger.error(f"Failed to read Excel {p}: {e}")
        return None

    def matched_count(self, true_predictions: List[Dict[str,Any]], gt_df: pd.DataFrame) -> int:
        if "outcome_id" not in gt_df.columns:
            logger.warning("GT missing outcome_id column, cannot align")
            return 0
        pred_df = pd.DataFrame(true_predictions)
        if pred_df.empty:
            return 0
        merged = gt_df.merge(pred_df[["outcome_id"]].drop_duplicates(), on="outcome_id", how="inner")
        return int(len(merged))


def print_table(table: Dict[str, Dict[str,int]]):
    header = f'{"Type":<38}{"Total":>10}{"Classifier SW":>14}{"TRUE Count":>10}'
    print("\n" + header)
    print("-" * len(header))
    order = [
        "ARTISTIC_AND_CREATIVE_PRODUCT",
        "KEY_FINDINGS",
        "RESEARCH_DATABASES_AND_MODELS",
        "SOFTWARE_AND_TECHNICAL_PRODUCT",
        "RESEARCH_TOOLS_AND_METHODS",
        "ENGAGEMENT_ACTIVITY",
        "INTELLECTUAL_PROPERTY"
    ]
    printed = set()
    for k in order:
        if k in table:
            v = table[k]
            print(f"{k:<38}{v['Total']:>10}{v['Classifier Software Count']:>14}{v['TRUE Count']:>10}")
            printed.add(k)
    for k, v in table.items():
        if k not in printed:
            print(f"{k:<38}{v['Total']:>10}{v['Classifier Software Count']:>14}{v['TRUE Count']:>10}")


def main():
    try:
        analyzer = AnalyzerConfirmedLogic()
        res = analyzer.run(limit=80000)

        # 1) Print statistics table
        print_table(res["table"])

        # 2) Only return alignment count
        gt = analyzer.load_gt()
        if gt is not None:
            matched = analyzer.matched_count(res["true_predictions"], gt)
            print(f"\n[Aligned with Ground Truth by outcome_id] matched_count = {matched}")
        else:
            print("\n[Alignment result] software.xlsx not found")

        # 3) Save JSON
        out_json = f"gtr_confirmed_logic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved: {out_json}")
    except Exception as e:
        print("Error occurred:", e)
        traceback.print_exc()


if __name__ == "__main__":
    main()