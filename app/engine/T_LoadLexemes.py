from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Iterable
import json
import os
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"


from .T_LoadMorphemes import AbstractMorpheme, am

@dataclass(slots=True)
class Lexeme:
    Morphemes: List["AbstractMorpheme"]
    Meaning: str = ""
    Category: Optional[str] = None
    name: str = ""
    Gender: Optional[str] = None
    Groups: List[str] = field(init=False, default_factory=list)
    MorphemeCategories: Dict[str, List[int]] = field(init=False, default_factory=dict)
    MorphemeGroups: Dict[str, List[int]] = field(init=False, default_factory=dict)
    DictionaryForm: Optional[str] = None

    def __post_init__(self):
        if self.Category is None and self.Morphemes:
            self.Category = self.Morphemes[-1].Category
            if self.Category == "causativity":
                self.Category = "verb"
            elif self.Category == "case":
                self.Category = "indeclinable"
    
        if (self.Meaning == "" or self.Meaning is None) and len(self.Morphemes) == 1:
            self.Meaning = getattr(self.Morphemes[-1], "meaning", "") or ""
    
        if self.Gender is None and self.Morphemes:
            self.Gender = getattr(self.Morphemes[-1], "Gender", "")
    
        if self.Morphemes:
            last = self.Morphemes[-1]
            if getattr(last, "Category", None) == "causativity" and len(self.Morphemes) >= 2:
                last = self.Morphemes[-2]
            if not hasattr(self, "Groups") or self.Groups is None:
                self.Groups = []
            self.Groups.extend(getattr(last, "Groups", []))

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "Meaning": self.Meaning,
            "Category": self.Category,
            "Gender": self.Gender,
            "Morphemes": [getattr(m, "name", None) for m in self.Morphemes],
            "Groups": self.Groups,
            "MorphemeCategories": self.MorphemeCategories,
            "MorphemeGroups": self.MorphemeGroups,
            "DictionaryForm": self.DictionaryForm
        }

    def clone(self) -> "Lexeme":
        new = object.__new__(Lexeme)  
        new.Meaning  = self.Meaning
        new.Category = self.Category
        new.name     = self.name
        new.Gender   = self.Gender
        new.DictionaryForm = self.DictionaryForm
        new.Morphemes     = list(self.Morphemes)  
        new.Groups        = list(self.Groups)
        new.MorphemeCategories = {k: list(v) for k, v in self.MorphemeCategories.items()}
        new.MorphemeGroups     = {k: list(v) for k, v in self.MorphemeGroups.items()}
        return new

    def __deepcopy__(self, memo):
        c = self.clone()
        memo[id(self)] = c
        return c

def _safe_get(am_dict: Dict[str, "AbstractMorpheme"], key: str):
    return am_dict.get(key)


def _unique_name(base: str, existing: Dict[str, Any]) -> str:
    if base not in existing:
        return base
    i = 1
    while True:
        cand = f"{base}{i}"
        if cand not in existing:
            return cand
        i += 1

def is_root(m):
    return m.name[0].islower()

def build_lexemes_from_am(
    am: Dict[str, "AbstractMorpheme"],
    json_path: Path = DATA_DIR / "lexemes.json",
    force_create: bool = False,
) -> Dict[str, Lexeme]:

    if (not force_create) and os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                print(f"Loaded {len(raw)} lexemes from json '{json_path}'.")
                # Reconstruct Lexeme objects from json
                lx: Dict[str, Lexeme] = {}
                for name, data in raw.items():
                    morphemes = [am[m] for m in (data.get("Morphemes") or []) if m in am]
                    lex = Lexeme(Morphemes=morphemes)
                    lex.name = data.get("name", name)
                    lex.Meaning = data.get("Meaning", "")
                    lex.Category = data.get("Category")
                    lex.Gender = data.get("Gender")
                    lex.Groups = data.get("Groups") or []
                    lex.MorphemeCategories = data.get("MorphemeCategories") or {}
                    lex.MorphemeGroups = data.get("MorphemeGroups") or {}
                    lex.DictionaryForm = data.get("DictionaryForm")
                    lx[name] = lex
                return lx
            else:
                print("Warning: json does not contain dict, recreating.")
        except Exception as e:
            print(f"Warning: failed to load json '{json_path}': {e}. Recreating lexemes.")

    lx: Dict[str, Lexeme] = {}

    for key, morph in am.items():
        try:
            if morph is None:
                continue

            if not is_root(morph):
                if key not in ["1","2","3","LOC","ABL","ALL","INS"]:
                    continue

            cat = getattr(morph, "Category", None)
            if cat is None:
                continue

            if str(cat).lower() == "verb":
                caus = getattr(morph, "Causativity", None)
                if caus is None:
                    pass
                else:
                    cs = str(caus)
                    cs_up = cs.upper()
                    if "K" in cs_up:
                        addon = _safe_get(am, "K")
                        if addon is not None:
                            lex = Lexeme(Morphemes=[morph, addon])
                            lex.name = _unique_name(f"K_{morph.name}", lx)
                            if cs == "VK":
                                lex.Meaning = f"{getattr(morph,'meaning', '')} (K)"
                            else:
                                lex.Meaning = f"{getattr(morph,'meaning', '')}"
                            lx[lex.name] = lex
                        else:
                            lex = Lexeme(Morphemes=[morph])
                            lex.name = _unique_name(f"K_{morph.name}", lx)
                            if cs == "VK":
                                lex.Meaning = f"{getattr(morph,'meaning','')} (K) [missing K morpheme]"
                            else:
                                lex.Meaning = f"{getattr(morph,'meaning', '')}"
                            lx[lex.name] = lex

                    if "V" in cs_up:
                        addon = _safe_get(am, "V")
                        if addon is not None:
                            lex = Lexeme(Morphemes=[morph, addon])
                            lex.name = _unique_name(f"V_{morph.name}", lx)
                            lex.Meaning = getattr(morph, "meaning", "")
                            lx[lex.name] = lex
                        else:
                            lex = Lexeme(Morphemes=[morph])
                            lex.name = _unique_name(f"V_{morph.name}", lx)
                            lex.Meaning = f"{getattr(morph,'meaning','')} (V) [missing V morpheme]"
                            lx[lex.name] = lex

                    if cs_up:
                        continue

            lex = Lexeme(Morphemes=[morph])
            cat_str = str(cat)
            if cat_str.lower() == "name":
                pref = "A"
            elif cat_str.lower() == "numberless":
                pref = "X"
            elif key in ["1","2","3"]:
                pref = "P"
            elif key in ["LOC","ABL","ALL","INS"]:
                pref = "I"
            elif cat_str:
                pref = cat_str[0].upper()
            else:
                pref = "X"
            lex.name = _unique_name(f"{pref}_{morph.name}", lx)
            lx[lex.name] = lex

        except Exception as e:
            print(f"Error while processing morpheme '{key}': {e}")

    try:
        lx_jsonable = {name: lex.to_jsonable() for name, lex in lx.items()}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(lx_jsonable, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(lx)} lexemes to json '{json_path}'.")
    except Exception as e:
        print(f"Error saving lexemes json '{json_path}': {e}")

    return lx


def _parse_groups(cell: Optional[str]) -> List[str]:
    if cell is None:
        return []
    s = str(cell).strip()
    if s == "":
        return []
    parts = [p.strip() for p in s.split(';') if p.strip() != ""]
    return parts

def load_extra_lexemes_from_excel(
    excel_path: str,
    am: Dict[str, Any],
    lx: Dict[str, Any],
    json_path: Path = DATA_DIR / "lexemes.json",
    sheet_name=0,
    name_col=0,
    meaning_col=1,
    morph_cols=(2, 3, 4),
    gender_col=5,
    groups_col=6, 
    overwrite_existing: bool = True
) -> Dict[str, Any]:

    first_letter_category = {
        "C": "creature",
        "D": "double",
        "I": "indeclinable",
        "M": "middle",
        "N": "numeral",
        "P": "person",
        "Q": "qualitative",
        "R": "relative",
        "S": "strong",
        "W": "weak",
        "B": "construct",
        "V": "verb",
        "X": "numberless"
    }

    def _norm_key_lookup_in_dict(d: Dict[str, Any], key: str):
        if key in d:
            return d[key]
        key_norm = key.strip().lower()
        for k, v in d.items():
            if k is None:
                continue
            try:
                if str(k).strip().lower() == key_norm:
                    return v
            except Exception:
                continue
        return None

    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    rows_by_name: Dict[str, Dict[str, Any]] = {}
    for idx, row in df.iterrows():
        if idx == 0:
            continue
        raw_name = row.iloc[name_col]
        if pd.isna(raw_name):
            continue
        name = str(raw_name).strip()
        if name == "":
            continue
        meaning = None if pd.isna(row.iloc[meaning_col]) else str(row.iloc[meaning_col]).strip()
        gender = None if pd.isna(row.iloc[gender_col]) else str(row.iloc[gender_col]).strip()
        morph_cells: List[Optional[str]] = []
        for c in morph_cols:
            try:
                val = row.iloc[c]
                morph_cells.append(None if pd.isna(val) else str(val).strip())
            except Exception:
                morph_cells.append(None)

        groups_list: List[str] = []
        if groups_col is not None:
            try:
                gval = row.iloc[groups_col]
                groups_list = _parse_groups(None if pd.isna(gval) else str(gval))
            except Exception:
                groups_list = []

        rows_by_name[name] = {
            "meaning": meaning,
            "morph_cells": morph_cells,
            "gender": gender,
            "Groups": groups_list,
            "row_index": idx
        }

    visiting = set()

    def _is_lexeme_ref_token(token: str) -> bool:
        if not token:
            return False
        token = str(token)
        if len(token) >= 2 and token[0].isupper() and token[1] == "_":
            return True
        return False

    def build_lexeme(name: str) -> Optional[Any]:
        if name in lx and not overwrite_existing:
            return lx[name]
        if name in visiting:
            print(f"Error: detected cyclic dependency while building lexeme '{name}'. Skipping.")
            return None

        row = rows_by_name.get(name)
        if row is None:
            if name in lx:
                return lx[name]
            print(f"Error: lexeme '{name}' referenced in excel but not found in sheet and not present in lx.")
            return None

        visiting.add(name)
        try:
            meaning = row["meaning"] or ""
            morph_cells = row["morph_cells"]
    
            accum_morphs: List[Any] = []
            accum_cats: Dict[str, List[int]] = {}
            accum_groups: Dict[str, List[int]] = {}
    
            def _merge_map(dst: Dict[str, List[int]], src: Dict[str, List[int]]):
                for k, idxs in (src or {}).items():
                    if not k: 
                        continue
                    dst.setdefault(k, []).extend(list(idxs))
    
            for ref in morph_cells:
                if ref is None or str(ref).strip() == "":
                    continue
                ref_str = str(ref).strip()
    
                if _is_lexeme_ref_token(ref_str):
                    base_lex = (lx.get(ref_str) or
                                (build_lexeme(ref_str) if ref_str in rows_by_name else _norm_key_lookup_in_dict(lx, ref_str)))
                    if base_lex is None:
                        print(f"Error: referenced lexeme '{ref_str}' for '{name}' not found.")
                        visiting.discard(name); return None
    
                    accum_morphs.extend(base_lex.Morphemes)
                    _merge_map(accum_cats,   base_lex.MorphemeCategories)
                    _merge_map(accum_groups, base_lex.MorphemeGroups)
    
                else:
                    am_obj = _norm_key_lookup_in_dict(am, ref_str)
                    if am_obj is None:
                        print(f"Error: referenced morpheme '{ref_str}' for lexeme '{name}' not found in am.")
                        visiting.discard(name); return None
                    accum_morphs.append(am_obj)
    
            lex = Lexeme(Morphemes=list(accum_morphs))
            lex.Meaning = meaning or ""
    
            first_letter = name[0] if name else ""
            cat = first_letter_category.get(first_letter.upper()) if first_letter else None
            lex.Category = cat if cat is not None else None
            lex.name = name
    
            gender = row.get("gender")
            if gender is not None and hasattr(lex, "Gender"):
                lex.Gender = gender
    
            groups_for_row = row.get("Groups", []) or []
            if groups_for_row:
                base = set(getattr(lex, "Groups", []) or [])
                base.update(groups_for_row)
                lex.Groups = list(base)
    
            lex.MorphemeCategories = {k: list(v) for k, v in accum_cats.items()}
            lex.MorphemeGroups     = {k: list(v) for k, v in accum_groups.items()}
    
            if lex.Morphemes:
                last_idx = len(lex.Morphemes) - 1
                if (lex.Category or "").strip():
                    lex.MorphemeCategories.setdefault(lex.Category, []).append(last_idx)
                for g in lex.Groups:
                    g = (g or "").strip()
                    if g:
                        lex.MorphemeGroups.setdefault(g, []).append(last_idx)
    
            if name in lx:
                print(f"Info: overwriting existing lexeme '{name}' in lx.")
            lx[name] = lex
            return lex
    
        finally:
            visiting.discard(name)

    for name in list(rows_by_name.keys()):
        if name in lx and not overwrite_existing:
            continue
        built = build_lexeme(name)
        if built is None:
            print(f"Warning: lexeme '{name}' could not be built from excel (see errors).")

    try:
        lx_jsonable = {}
        for k, v in lx.items():
            if hasattr(v, "to_jsonable"):
                lx_jsonable[k] = v.to_jsonable()
            else:
                lx_jsonable[k] = {
                    "name": getattr(v, "name", k),
                    "Meaning": getattr(v, "Meaning", "") if hasattr(v, "Meaning") else "",
                    "Category": getattr(v, "Category", None),
                    "Morphemes": [getattr(m, "name", None) for m in getattr(v, "Morphemes", [])]
                }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(lx_jsonable, f, ensure_ascii=False, indent=2)
        print(f"Saved updated lexemes to json '{json_path}'.")
    except Exception as e:
        print(f"Error saving json '{json_path}': {e}")

    return lx

def get_or_build_lexemes(
    am: Dict[str, Any],
    lexeme_json: Path = DATA_DIR / "lexemes.json",
    excel_extra_path: Optional[str] = None,
    excel_sheet: int = 0,
    groups_col: int = 6,
    force_create: bool = False,
    overwrite_existing: bool = True,
    morpheme_builder_kwargs: Optional[dict] = None,
    extra_loader_kwargs: Optional[dict] = None,
) -> Dict[str, Any]:

    if (not force_create) and os.path.exists(lexeme_json):
        try:
            with open(lexeme_json, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                lx: Dict[str, Any] = {}
                for name, data in raw.items():
                    morphemes = [am[m] for m in (data.get("Morphemes") or []) if m in am]
                    lex = Lexeme(Morphemes=morphemes)
                    lex.name = data.get("name", name)
                    lex.Meaning = data.get("Meaning", "")
                    lex.Category = data.get("Category")
                    lex.Gender = data.get("Gender")
                    lex.Groups = data.get("Groups") or []
                    lex.MorphemeCategories = data.get("MorphemeCategories") or {}
                    lex.MorphemeGroups = data.get("MorphemeGroups") or {}
                    lex.DictionaryForm = data.get("DictionaryForm")
                    lx[name] = lex
                # print(f"Loaded {len(lx)} lexemes from json '{lexeme_json}'.")
                return lx
        except Exception as e:
            print(f"Warning: failed to load lexemes from json '{lexeme_json}': {e}. Will rebuild.")

    mb_kwargs = morpheme_builder_kwargs.copy() if morpheme_builder_kwargs else {}
    mb_kwargs.setdefault("json_path", lexeme_json)
    mb_kwargs.setdefault("force_create", True)

    lx = build_lexemes_from_am(am, **mb_kwargs)
    if lx is None:
        lx = {}

    if excel_extra_path:
        el_kwargs = extra_loader_kwargs.copy() if extra_loader_kwargs else {}
        el_kwargs.setdefault("am", am)
        el_kwargs.setdefault("lx", lx)
        el_kwargs.setdefault("json_path", lexeme_json)
        el_kwargs.setdefault("sheet_name", excel_sheet)
        el_kwargs.setdefault("groups_col", groups_col)
        el_kwargs.setdefault("overwrite_existing", overwrite_existing)

        result = load_extra_lexemes_from_excel(excel_extra_path, **el_kwargs)
        if result is not None:
            lx = result

    try:
        lx_jsonable = {}
        for name, lex in lx.items():
            if hasattr(lex, "to_jsonable"):
                lx_jsonable[name] = lex.to_jsonable()
            else:
                lx_jsonable[name] = {
                    "name": getattr(lex, "name", name),
                    "Meaning": getattr(lex, "Meaning", "") if hasattr(lex, "Meaning") else getattr(lex, "Meaning", ""),
                    "Category": getattr(lex, "Category", None),
                    "Morphemes": [getattr(m, "name", None) for m in getattr(lex, "Morphemes", [])],
                    "Gender": getattr(lex, "Gender", "") if hasattr(lex, "Gender") else getattr(lex, "Gender", ""),
                    **({"Groups": getattr(lex, "Groups")} if hasattr(lex, "Groups") else {}),
                }
        with open(lexeme_json, "w", encoding="utf-8") as f:
            json.dump(lx_jsonable, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(lx)} lexemes to json '{lexeme_json}'.")
    except Exception as e:
        print(f"Warning: failed to save lexemes json '{lexeme_json}': {e}")

    return lx

lx = get_or_build_lexemes(
    am,
    lexeme_json=DATA_DIR / "lexemes.json",
    excel_extra_path=DATA_DIR / "lexemes.xlsx",
    excel_sheet=0,
    groups_col=6,
    force_create=False,
    overwrite_existing=True
)