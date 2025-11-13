from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
import pandas as pd
import pickle
import json
import os
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"


@dataclass(slots=True, frozen=True)
class AbstractMorpheme:
    name: str
    shape: Optional[str] = None
    meaning: Optional[str] = None
    Category: Optional[str] = None
    Color: Optional[str] = None
    Transitivity: Optional[str] = None
    Voice: Optional[str] = None
    Causativity: Optional[str] = None
    Gender: Optional[str] = None
    Language: Optional[str] = None
    Groups: Optional[List[str]] = None

    def fast_clone_context(self) -> "ContextMorpheme":
            return ContextMorpheme.from_abstract_fast(self)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d
    
    def __repr__(self):
        return self.name

@dataclass(slots=True)
class Letter:
    morpheme: "ContextMorpheme"
    shape: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    def __repr__(self):
        return self.shape


@dataclass(slots=True)
class ContextMorpheme:
    shape: Optional[str] = None
    meaning: Optional[str] = None
    Category: Optional[str] = None
    Color: Optional[str] = None
    Transitivity: Optional[str] = None
    Voice: Optional[str] = None
    Causativity: Optional[str] = None
    Gender: Optional[str] = None
    Language: Optional[str] = None
    Groups: Optional[List[str]] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    wagon: int = 2
    name: str = ""
    
    def __repr__(self):
        return self.name
    
    def __eq__(self,other):
        if isinstance(other, AbstractMorpheme):
            return self.name == other.name
        if isinstance(other, ContextMorpheme):
            return self.id == other.id
        if isinstance(other, Letter):
            return self.id == other.morpheme.id
        if isinstance(other,str):
            return self.name == other
        return False
        

    @classmethod
    def from_abstract_fast(cls, am: AbstractMorpheme) -> "ContextMorpheme":

        if am.Groups is None:
            groups_copy = None
        else:
            groups_copy = list(am.Groups)

        return cls(
            name=am.name,
            shape=am.shape,
            meaning=am.meaning,
            Category=am.Category,
            Color=am.Color,
            Transitivity=am.Transitivity,
            Voice=am.Voice,
            Causativity=am.Causativity,
            Gender=am.Gender,
            Language=am.Language,
            Groups=groups_copy
        )


def _clean_cell(val: Any) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s if s != "" else None

def _parse_groups(cell_val: Any) -> List[str]:
    if pd.isna(cell_val):
        return []
    s = str(cell_val).strip()
    if s == "":
        return []
    parts = [p.strip() for p in s.split(';') if p.strip() != ""]
    return parts

def load_or_create_morphemes(
    excel_path: Path = DATA_DIR / "morphemes.xlsx",
    pickle_path: Path = DATA_DIR / "morphemes.pickle",
    json_path: Path = DATA_DIR / "morphemes.json",
    force_create: bool = False
) -> Dict[str, AbstractMorpheme]:
    """
    Если pickle_path существует и force_create == False -> загружает am из pickle и возвращает.
    Иначе читает excel_path, создаёт морфемы и сохраняет в pickle_path и json_path, затем возвращает.
    """

    if (not force_create) and os.path.exists(pickle_path):
        try:
            with open(pickle_path, "rb") as f:
                am = pickle.load(f)
            if isinstance(am, dict):
                # print(f"Loaded {len(am)} morphemes from pickle '{pickle_path}'.")
                return am
            else:
                print(f"Warning: pickle '{pickle_path}' не содержит dict, пересоздаём из excel.")
        except Exception as e:
            print(f"Warning: не удалось загрузить pickle '{pickle_path}': {e}. Пересоздаём из excel.")

    xls = pd.read_excel(excel_path, sheet_name=None, header=0)
    # xls — OrderedDict {sheet_name: DataFrame}

    entries = []
    sheet_names = list(xls.keys())

    for sheet_index, sheet_name in enumerate(sheet_names):
        df = xls[sheet_name]

        ncols = df.shape[1]
        if ncols < 3:
            continue

        for ri, row in df.iterrows():
            if row.isnull().all():
                continue

            shape = _clean_cell(row.iloc[1])  
            meaning = _clean_cell(row.iloc[2])  


            props = {}
            for colname in ["Category", "Gender", "Language", "Groups", "Color",
                            "Transitivity","Voice","Causativity"]:
                if colname in df.columns:
                    val = row[colname]
                    if colname == "Groups":
                        props["Groups"] = _parse_groups(val)
                    else:
                        props[colname] = _clean_cell(val)
                else:
                    props.setdefault(colname, None)

            if sheet_index <= 1:
                name_raw = meaning if meaning is not None else (shape or "")
            else:
                name_raw = shape if shape is not None else (meaning or "")

            entry = {
                "sheet_index": sheet_index,
                "sheet_name": sheet_name,
                "shape": shape,
                "meaning": meaning,
                "raw_name": name_raw,
                "props": props
            }
            if entry["raw_name"] == "" or entry["raw_name"] is None:
                entry["raw_name"] = f"__unnamed__sheet{sheet_index}_row{ri}"
            entries.append(entry)

    name_counts: Dict[str, int] = {}
    for e in entries:
        rn = e["raw_name"]
        name_counts[rn] = name_counts.get(rn, 0) + 1

    name_index: Dict[str, int] = {}

    am: Dict[str, AbstractMorpheme] = {}
    for e in entries:
        raw = e["raw_name"]
        count = name_counts.get(raw, 0)
        name_index[raw] = name_index.get(raw, 0) + 1
        idx = name_index[raw]

        if count > 1:
            final_name = f"{raw}{idx}"
        else:
            final_name = raw

        if final_name in am:
            j = 1
            while f"{final_name}_{j}" in am:
                j += 1
            final_name = f"{final_name}_{j}"

        p = e["props"]
        groups = p.get("Groups") or []

        m = AbstractMorpheme(
            name=final_name,
            shape=e["shape"],
            meaning=e["meaning"],
            Category=p.get("Category"),
            Gender=p.get("Gender"),
            Language=p.get("Language"),
            Color=p.get("Color"),
            Transitivity=p.get("Transitivity"),
            Voice=p.get("Voice"),
            Causativity=p.get("Causativity"),
            Groups=groups
        )
        am[final_name] = m

    print(f"Created {len(am)} morphemes from Excel '{excel_path}' (sheets: {len(sheet_names)}).")

    try:
        with open(pickle_path, "wb") as f:
            pickle.dump(am, f)
        print(f"Saved pickle to '{pickle_path}'.")
    except Exception as e:
        print(f"Error saving pickle '{pickle_path}': {e}")

    try:
        am_jsonable = {name: obj.to_dict() for name, obj in am.items()}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(am_jsonable, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON to '{json_path}'.")
    except Exception as e:
        print(f"Error saving JSON '{json_path}': {e}")

    try:
        with open(pickle_path, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict):
            print(f"Verified pickle by re-loading: {len(loaded)} items.")
            return loaded
        else:
            print("Pickle verification: содержимое не dict, возвращаем созданный am.")
            return am
    except Exception as e:
        print(f"Не удалось перезагрузить pickle после сохранения: {e}. Возвращаем созданный am.")
        return am


excel_file = DATA_DIR / "morphemes.xlsx"
pickle_file = DATA_DIR / "morphemes.pickle"
json_file = DATA_DIR / "morphemes.json"

am = load_or_create_morphemes(excel_file, pickle_file, json_file, force_create=False)