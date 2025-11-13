from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import pandas as pd
import pickle
import json
import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

def clean_cell(cell):

    if pd.isna(cell):
        return None
    if not isinstance(cell, str):
        return cell
    s = cell.strip()
    if s == "":
        return None
    s = _leading_quote_re.sub("", s)
    s = s.strip()
    return s if s != "" else None


@dataclass(slots=True)
class Rule:
    id: str
    kind: str
    ifs: List[str]
    unlesses: List[str]
    operations: List[str]
    optional: List[str]
    wording: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def _split_cell(cell: Optional[str]) -> List[str]:
    """
    Split cell by ';', strip items and drop empties.
    Accepts None/NaN.
    """
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if s == "":
        return []
    parts = [p.strip() for p in s.split(";")]
    return [p for p in parts if p != ""]

def _unique_id(base_id: str, existing: Dict[str, Any]) -> str:
    """
    Если base_id уже в existing — добавляет суффикс _1, _2, ... для уникальности.
    """
    if base_id not in existing:
        return base_id
    i = 1
    while True:
        cand = f"{base_id}_{i}"
        if cand not in existing:
            return cand
        i += 1

_leading_quote_re = re.compile(r"^[\s\uFEFF\u00A0]*['\u2019\u2018\`]+\s*")  

def load_rules_from_excel(
    excel_path: Path = DATA_DIR /  "rules.xlsx",
    pickle_path: Path = DATA_DIR /  "rules.pickle",
    json_path: Path = DATA_DIR /  "rules.json",
    sheet_name: int = 0,
    type_col: int = 0,
    id_col: int = 1,
    if_col: int = 2,
    unless_col: int = 3,
    then_col: int = 4,
    optional_col: int = 5,
    wording_col: int = 6,
    skip_header: bool = True,
    force_create: bool = False
) -> Dict[str, Rule]:

    if (not force_create) and os.path.exists(pickle_path):
        try:
            with open(pickle_path, "rb") as f:
                rules_loaded = pickle.load(f)
            return rules_loaded
        except Exception as e:
            print(f"Warning: failed to load rules from pickle '{pickle_path}': {e}. Rebuilding from Excel.")

    # Reading excel
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    except Exception as e:
        raise RuntimeError(f"Cannot read excel '{excel_path}': {e}")

    if skip_header:
        df_iter = df.iloc[1:].itertuples(index=False)
    else:
        df_iter = df.itertuples(index=False)

    rules: Dict[str, Rule] = {}

    for row in df_iter:
        try:
            raw_id = row[id_col]
            type_id = row[type_col]
            try:
                raw_id = row[id_col]
            except Exception:
                raw_id = None
            if raw_id is None:
                continue
            rule_id = str(raw_id).strip()
            if rule_id == "":
                continue

            if type_id == "M":
                kind = "morphological"
            elif type_id == "R":
                kind = "phonological_right"
            else:
                kind = "phonological"

            def _get_cell(row, idx):
                try:
                    val = row[idx]
                except Exception:
                    return None
                return clean_cell(val)

            if_cell = _get_cell(row, if_col)
            unless_cell = _get_cell(row, unless_col)
            then_cell = _get_cell(row, then_col)
            opt_cell = _get_cell(row, optional_col)
            wording_cell = _get_cell(row, wording_col)

            ifs = _split_cell(if_cell)
            unlesses = _split_cell(unless_cell)
            operations = _split_cell(then_cell)
            optional = _split_cell(opt_cell)
            wording = _split_cell(wording_cell)[0]

            uid = _unique_id(rule_id, rules)
            if uid != rule_id:
                print(f"Warning: duplicate rule id '{rule_id}' -> storing as '{uid}'")

            r = Rule(
                id=uid,
                kind=kind,
                ifs=ifs,
                unlesses=unlesses,
                operations=operations,
                optional=optional,
                wording=wording
            )
            rules[uid] = r

        except Exception as e:
            print(f"Error parsing row: {e}. Row data: {row}")

    try:
        with open(pickle_path, "wb") as f:
            pickle.dump(rules, f)
        print(f"Saved {len(rules)} rules to pickle '{pickle_path}'.")
    except Exception as e:
        print(f"Warning: failed to save rules pickle '{pickle_path}': {e}")

    try:
        rules_jsonable = {rid: rule.to_dict() for rid, rule in rules.items()}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rules_jsonable, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(rules)} rules to json '{json_path}'.")
    except Exception as e:
        print(f"Warning: failed to save rules json '{json_path}': {e}")

    return rules

rules = load_rules_from_excel(skip_header=True)