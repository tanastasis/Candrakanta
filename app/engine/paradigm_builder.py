import json
import hashlib
import unicodedata
from datetime import datetime, timezone
from .T1 import (
    Form, Variant, showvar, lx, am,
    Grammeme, Postfixeme, phonol, buildForm, standard, sublexeme,
    Lexeme, dictionary_form, rules, show_form
)

from typing import Dict, List, Tuple
from pprint import pprint

def _norm_surface(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u00A0", " ")    # NBSP -> space
    s = s.replace("\u200B", "")     # zero-width space -> remove
    return s

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def make_form_id(lexeme_id: str, grammemes: list[str], postfixemes: list[str]) -> str:
    g = ",".join(sorted(grammemes))
    p = ",".join(sorted(postfixemes))
    raw = f"{lexeme_id}||g=[{g}]||p=[{p}]"
    return _sha1(raw)

def make_variant_id(form_id: str, variant_signature: str) -> str:
    raw = f"{form_id}||{variant_signature}"
    return _sha1(raw)

# --- form/variatn serialization ---

def variant_signature_from_steps(variant) -> str:
    # 1) surface
    try:
        surface = showvar(variant)  # твой метод
    except Exception:
        # fallback: склейка букв
        surface = "".join(getattr(let, "shape", "") for let in getattr(variant, "l", []))
    surface = _norm_surface(surface)

    # 2) шаги (если есть)
    steps = []
    rules = getattr(variant, "rules", None)  # если ты писал такой атрибут
    if rules:
        # ожидаем элементы из твоего лога: (rule_id, position, optional_flag, gloss)
        for r in rules:
            rule_id = str(r[0])
            pos = "" if r[1] is None else str(r[1])
            opt = "1" if (len(r) > 2 and r[2]) else "0"
            steps.append(f"{rule_id}@{pos}:{opt}")

    # если шагов нет — оставим только surface
    sig = f"surface={surface}"
    if steps:
        sig += "||steps=" + "|".join(steps)
    return sig

def variant_to_json(variant, form_id: str) -> tuple[dict, dict]:
    surface = _norm_surface(showvar(variant))
    signature = variant_signature_from_steps(variant)
    vid = make_variant_id(form_id, signature)

    rules = getattr(variant, "rules", None) or []
    steps = []
    for r in rules:
        rule_id = str(r[0])
        pos = r[1]
        opt = r[2] if len(r) > 2 else ""
        form_after = r[3] if len(r) > 3 else ""
        steps.append({"rule_id": rule_id, "pos": pos, "opt": opt, "form": form_after})

    variant_json = {
        "variant_id": vid,
        "surface": surface,
        "is_canonical": standard(variant),
    }
    derivation_payload = {
        "variant_id": vid,
        "form_id": form_id,
        "surface": surface,
        "steps": steps,
    }
    return variant_json, derivation_payload


def form_to_json(lexeme_id: str, grammemes: list[str], postfixemes: list[str], form_obj, cell_label: str = "—"):
    form_id = make_form_id(lexeme_id, grammemes, postfixemes)
    variants = getattr(form_obj, "Variants", None) or []
    variants_sorted = sorted(variants, key=lambda v: _norm_surface(showvar(v)))

    ui_variants = []
    derivations = []   # ← сюда сложим кэшируемые записи
    for v in variants_sorted:
        v_json, deriv = variant_to_json(v, form_id)
        ui_variants.append(v_json)
        derivations.append(deriv)

    return {"cell": cell_label, "form_id": form_id, "variants": ui_variants}, derivations

ROW_PERSONS = ["1", "2", "3"]
ROW_NUMBERS = ["SG", "PL"]
ROW_VOICES  = ["ACT", "MID"]
COL_TENSES  = ["PRS", "IPF", "CON", "OPT", "PRT", "IMP"]

SUBLEXEME_TAGS_BASE = ["PPA", "PPP", "GER1", "GER2", "INF", "PPM"]

def _make_row_id(person: str, number: str, voice: str) -> str:
    # e.g. "1.SG.ACT"
    return f"{person}.{number}.{voice}"

def _build_row_values() -> List[Dict[str, str]]:
    """
    Order: keep one voice block, inside it all numbers, inside numbers all persons.
    i.e., 1.SG.ACT, 2.SG.ACT, 3.SG.ACT, 1.PL.ACT, 2.PL.ACT, 3.PL.ACT, then MID block.
    """
    values: List[Dict[str, str]] = []
    for voice in ROW_VOICES:
        for number in ROW_NUMBERS:
            for person in ROW_PERSONS:
                row_id = _make_row_id(person, number, voice)
                values.append({"id": row_id, "label": row_id})
    return values

def export_paradigm_verb(lex: Lexeme, *, save_to: str | None = None) -> Dict:
    if getattr(lex, "Category", None) != "verb":
        raise ValueError(f"Lexeme {lex.name} is not 'verb'")

    axes = [
        {"id":"row","label":"PNV","values": _build_row_values()},
        {"id":"tense","label":"Tense","values":[{"id":t,"label":t} for t in COL_TENSES]},
    ]
    layout = {"rows":"row","cols":"tense","pages":None}

    cells, all_derivations = [], []

    # finite forms
    for voice in ROW_VOICES:
        for number in ROW_NUMBERS:
            for person in ROW_PERSONS:
                row_id = _make_row_id(person, number, voice)
                for tense in COL_TENSES:
                    if tense == "IMP" and person != "2":
                        continue
                    gr = Grammeme([am[tense], am[person], am[number], am[voice]])
                    form = phonol(buildForm(lex, gr))
                    cell_json, derivs = form_to_json(lex.name, [tense,person,number,voice], [], form)
                    
                    for d in derivs:
                        d["lexeme_id"] = lex.name
                        d["gr_obj"]    = gr
                    
                    all_derivations.extend(derivs)

                    cells.append({
                        "coords":{"row":row_id,"tense":tense},
                        "content":{"type":"form","form_id":cell_json["form_id"],"variants":cell_json["variants"]}
                    })

    # sublexeme rows (link to ?sub=TAG)
    sub_tags = list(SUBLEXEME_TAGS_BASE)

    for tag in sub_tags:
        row_id = f"{tag}"
        axes[0]["values"].append({"id":row_id,"label":row_id})

        sub = sublexeme(lex, [tag])
        label = dictionary_form(sub, every=True)

        cells.append({
            "coords":{"row":row_id,"tense":COL_TENSES[0]},
            "content":{
                "type":"sublexeme",
                "lexeme_ref":{"sub_tag": tag},
                "label": label
            }
        })

    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lexeme_id": lex.name,
        "meaning": lex.Meaning,
        "category": "verb",
        "axes": axes,
        "layout": layout,
        "cells": cells,
        "_derivations": all_derivations,
    }
    if save_to:
        with open(save_to, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload

# ---- predicates / helpers ----
def has_group(g):
    return lambda lex: g in set(getattr(lex, "Groups", []) or [])

def is_name_in(names):
    return lambda lex: (getattr(lex, "name", "") in names)

def parse_col(tok: str):
    # "M.SG" -> ("M","SG"), "M"->("M",None), "—"->(None,None)
    if tok == "—": return (None, None)
    p = tok.split(".")
    return (p[0], p[1]) if len(p) == 2 else (p[0], None)

def build_gram(case, gender=None, number=None, deixis=None):
    grams = []
    if gender: grams.append(am[gender])
    if number: grams.append(am[number])
    if deixis: grams.append(am[deixis])
    grams.append(am[case])
    return Grammeme(grams)

# ---- reusable grids ----
CASES_DEFAULT = ["NOM","ACC","GEN","INS","ALL","LOC","COM","PER","ABL"]
CASES_PLUS_OBL = ["NOM","ACC","GEN","INS","ALL","LOC","COM","PER","ABL","OBL"]
CASES_PLUS_ADV = ["NOM","ACC","GEN","INS","ALL","LOC","COM","PER","ABL","ABL3","ADV"]

# ---- category profiles ----
class Profile:
    def row_spec(self, lex): ...
    def col_spec(self, lex): ...
    def cell_rule(self, lex, case: str, col: str): ...
    def extra_rows(self, lex): return []

class PersonProfile(Profile):
    def row_spec(self, lex):
        if lex.name == "P_änzaṃ": return CASES_DEFAULT
        return CASES_PLUS_OBL
    def col_spec(self, lex):
        nm = getattr(lex, "name", "")
        if nm == "P_1": return ["M.SG","F.SG","PL"]
        if nm == "P_2": return ["SG","PL"]
        if nm == "P_änzaṃ": return ["—","M.SG","M.PL","F.SG","F.PL"]
    def cell_rule(self, lex, case, col):
        if case == "OBL" and "." in col:
            return None
        if col == "—" and case not in ["LOC","PER"]:
            return None            
        g, n = parse_col(col)
        if lex.name == "P_änzaṃ" and col != "—":
            gr = build_gram(case, g, n, "MED")
        else:
            gr = build_gram(case, g, n)
        return {"type":"form","gr":gr}
    def extra_rows(self, lex):
        rows = []
        # sublexemes
        if lex.name == "P_2":
            rows.append({"kind":"big_sublexeme", "tags":["SG","ADJ"]})
        return rows

class NumeralProfile(Profile):
    def row_spec(self, lex): return CASES_DEFAULT
    def col_spec(self, lex):
        nm = getattr(lex, "name", "")
        if nm == "N_ṣomă": return ["—","M.SG","M.PL","F.SG","F.PL"]
        if nm == "N_vä": return ["—","M","F"]
        if nm == "N_täräy": return ["M","F"]
        if nm == "N_ṣäpta": return ["—","PL"]
        return ["—"]
    def cell_rule(self, lex, case, col):
        if lex.name in ["N_vä","N_ṣomă"] and col == "—" and case != "NOM":
            return None
        g, n = parse_col(col)
        gr = build_gram(case, g, n)
        return {"type":"form","gr":gr}
    def extra_rows(self, lex):
        rows = []
        # dictionary form row
        rows.append({"kind":"form_fullrow", "label":"Construct form", "gr":Grammeme([])})
        # sublexemes
        rows.append({"kind":"sublexeme", "tag":"ORD"})
        if not lex.name.endswith(("PLUS","TEN","śäka")):
            rows.append({"kind":"sublexeme", "tag":"PLUS"})
            if lex.name != "N_ṣomă":
                rows.append({"kind":"sublexeme", "tag": "TEN"})
        return rows

class IndeclinableProfile(Profile):
    def row_spec(self, lex):
        if has_group("Ac")(lex):
            cases = []
        else:
            cases = ["NOM"]
        if lex.name in ["I_pos"]:
            cases.extend(["ALL","LOC","PER"])
        if lex.name in ["I_menāk"]:
            cases.extend(["INS"])
        if lex.name in ["I_ywār","I_el"]:
            cases.extend(["LOC","PER"])
        if lex.name == "I_ṣolār":
            cases.extend(["LOC"])
        if lex.name in ["I_neṣ","I_snepal"]:
            cases.extend(["PER"])
        if lex.name in ["I_antu","I_λutār","I_ś`äw"]:
            cases.append("ABL")
        if lex.name == "I_korp":
            cases.extend(["ALL","PER"])
        if has_group("Ṣurmă")(lex):
            cases.append("ABL2")
        if has_group("Anăpr")(lex):
            cases.append("ABL3")
            

        return cases
    def col_spec(self, lex):
        return ["—"]
    def cell_rule(self, lex, case, col):
        g, n = parse_col(col)
        gr = build_gram(case, g, n)
        return {"type":"form","gr":gr}
    def extra_rows(self, lex):
        rows = []
        if has_group("Ywār")(lex):
            rows.append({"kind":"form_fullrow", "label":"Construct form", "gr":Grammeme([])})
        if has_group("Ksär")(lex):
            rows.append({"kind":"sublexeme", "tag":"ADJ"})
        if has_group("Ri")(lex):
            rows.append({"kind":"sublexeme", "tag":"HAB"})
        if has_group("Riṣak")(lex):
            rows.append({"kind":"sublexeme", "tag":"NMZ"})
        if has_group("Kñuk")(lex):
            rows.append({"kind":"sublexeme", "tag":"BAH1"})
        return rows



class DoubleProfile(Profile):
    def row_spec(self, lex): return CASES_DEFAULT
    def col_spec(self, lex):
        if lex.name in ["D_esă","D_poke","D_pe"]:
            return ["SG","DU","PL"]
        return ["SG","DU","PL"]
    def cell_rule(self, lex, case, col):
        g, n = parse_col(col)
        gr = build_gram(case, g, n)
        return {"type":"form","gr":gr}
    def extra_rows(self, lex):
        rows = []
        if has_group("Kñuk")(lex):
            rows.append({"kind":"sublexeme", "tag":"BAH1"})
        if has_group("Ag")(lex):
            rows.append({"kind":"sublexeme", "tag":"BAH1"})
        if lex.name in ["D_kanwe","D_kloz","D_päśśă"]:
            rows.append({"kind":"sublexeme", "tag":"ADJ"})            
        return rows

class NounProfile(Profile):
    def row_spec(self, lex):
        cases = CASES_DEFAULT[:]
        if has_group("Ṣurmă")(lex):
            cases.append("ABL2")
        if has_group("Anăpr")(lex) or lex.name == "W_käλme":
            cases.append("ABL3")
        if has_group("Saṅkrām")(lex):
            cases.append("LOCAL")    
        return cases
    def col_spec(self, lex):
        if lex.name.endswith("a`"):
            return ["—","SG","PL"]
        if lex.name == "C_pracăr":
            return ["SG","DU","PL"]
        if lex.name == "X_qä":
            return ["—","PL"]
        if lex.Category in ["numberless","number"]:
            return ["—"]
        if lex.Category == "name" and lex.name not in ["A_śākyamuni","A_upendre"]:
            return ["SG"]
        return ["SG","PL"]
    def cell_rule(self, lex, case, col):
        if case == "ABL3" and lex.name == "W_käλme":
            if col != "PL":
                return None
        elif case in ["ABL2","ABL3"] and col == "PL":
            return None
        elif lex.name == "X_qä" and col == "PL" and case not in ["NOM","ACC"]:
            return None
        elif col == "—":
            if lex.name == "W_ṣurmă":
                if case not in ["ABL","ABL2"]:
                    return None
            elif lex.name == "M_särka":
                if case not in ["ABL2","ABL3"]:
                    return None
            elif lex.name == "X_ālamwäc":
                if case == "NOM":
                    return None
            elif case not in ["NOM"] and lex.Category not in ["numberless","number"]:
                return None
        g, n = parse_col(col)
        gr = build_gram(case, g, n)
        return {"type":"form","gr":gr}
    def extra_rows(self, lex):
        rows = []
        if lex.name in ["C_pättāñkät","C_wlāñkät","C_śriñkät"]:
            return rows
        if has_group("Ywār")(lex):
            rows.append({"kind":"form_fullrow", "label":"Construct form", "gr":Grammeme([])})
        if lex.Category != "numberless" or has_group("Ksär")(lex):
            rows.append({"kind":"sublexeme", "tag":"ADJ"})
        if has_group("Lānt")(lex):
            rows.append({"kind":"sublexeme", "tag":"F"})
        if has_group("Riṣak")(lex) or lex.name in ["S_lānt+F","S_wāpant+F"]:
            rows.append({"kind":"sublexeme", "tag":"NMZ"})
        if has_group("Asurā")(lex):
            rows.append({"kind":"sublexeme", "tag":"POS"})
        if has_group("Kñuk")(lex):
            rows.append({"kind":"sublexeme", "tag":"BAH1"})
        if has_group("Pälzäk")(lex):
            rows.append({"kind":"sublexeme", "tag":"BAH2"})
        if has_group("Ri")(lex):
            rows.append({"kind":"sublexeme", "tag":"HAB"})
        return rows

class AdjectiveProfile(Profile):
    def row_spec(self, lex):
        cases = CASES_DEFAULT[:]
        if lex.name in ["V_täm+PPP","Q_mkältāwr"]:
            cases.append("ABL3")
        return cases
    def col_spec(self, lex):
        if lex.name == "R_p":
            return ["—","M.SG","F.SG","DU","M.DU","F.DU","M.PL","F.PL"]
        if lex.Morphemes[-1].name in ["PPP","ālăk"] or (lex.Morphemes[-1].name == "ADJ" and lex.Morphemes[-2].name == "päqälā") \
            or lex.name in ["Q_mkältāwr","N_vä+ORD"]:
            return ["—","M.SG","F.SG","M.PL","F.PL"]
        if has_group("Ṣomă")(lex):
            return ["—","M.SG","F.SG","M.PL","F.PL"]
        return ["M.SG","F.SG","M.PL","F.PL"]
    def cell_rule(self, lex, case, col):
        if col == "—":
            if has_group("Ṣomă")(lex) and case != "NOM":
                return None
            if lex.Morphemes[-1].name in ["PPP","mkältāwr"]:
                if case in ["ABL","PER"] and lex.Morphemes[-1].name != "mkältāwr":
                    pass
                elif case == "ABL3" and lex.name in ["V_täm+PPP","Q_mkältāwr"]:
                    pass
                elif case in ["NOM","ACC"] and lex.name in ["V_tärk+PPP","V_kärs+PPP"]:
                    pass
                elif case == "INS" and lex.name in ["V_näzv+PPP","V_läm+PPP","V_λāw+PPP"]:
                    pass
                else:
                    return None
            if lex.name == "N_vä+ORD" and case != "PER":
                return None
            if (lex.Morphemes[-1].name == "ADJ" and lex.Morphemes[-2].name in "päqälā") and case not in ['ABL','ACC']:
                return None
        elif case == "ABL3":
            return None
        g, n = parse_col(col)
        gr = build_gram(case, g, n)
        return {"type":"form","gr":gr}
    def extra_rows(self, lex):
        rows = []
        if has_group("Ywār")(lex):
            rows.append({"kind":"form_fullrow", "label":"Construct form", "gr":Grammeme([])})
        if has_group("Ksär")(lex):
            rows.append({"kind":"sublexeme", "tag":"ADJ"})
        if lex.Category == "qualitative" or \
            has_group("Riṣak")(lex) or lex.name in ["V_knās+PPM","V_knās+PPM+NMZ+BAH1"]:
                rows.append({"kind":"sublexeme", "tag":"NMZ"})
                return rows

# профили по категориям
PROFILES = {
    "numeral": NumeralProfile(),
    "person": PersonProfile(),
    "indeclinable": IndeclinableProfile(),
    "double": DoubleProfile(),
    "relative": AdjectiveProfile(),
    "qualitative": AdjectiveProfile()
}
DEFAULT_PROFILE = NounProfile()

# ---- engine ----
def export_paradigm_nominal(lex, *, save_to: str|None=None):
    if has_group("Prattika")(lex):
        form0 = phonol(buildForm(lex, Grammeme([])))
        cell0, derivs0 = form_to_json(
            getattr(lex, "name", "LEX"),
            grammemes=[],
            postfixemes=[],
            form_obj=form0,
            cell_label=None,
        )
        payload = {
            "version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "lexeme_id": getattr(lex, "name", "LEX"),
            "category": getattr(lex, "Category", "") or "",
            "meaning": lex.Meaning,
            "axes": [
                {"id":"case","label":"Case","values":[{"id":"—","label":"—"}]},
                {"id":"col", "label":"Gender/Number","values":[{"id":"—","label":"—"}]},
            ],
            "layout": {"rows":"case","cols":"col","pages":None},
            "cells": [{
                "coords": {"case":"—","col":"—"},
                "content": {
                    "type":"form",
                    "form_id": cell0["form_id"],
                    "variants": cell0["variants"],
                }
            }],
            "_derivations": derivs0,
        }
        return payload    

    category = (getattr(lex, "Category", "") or "").lower()
    profile = PROFILES.get(category, DEFAULT_PROFILE)

    cases = profile.row_spec(lex)
    cols  = profile.col_spec(lex)
    print(f"DEBUG: lex.name={lex.name}, Groups={getattr(lex, 'Groups', [])}, cols={cols}")
    

    axes = [
        {"id":"case","label":"Case","values":[{"id":c,"label":c} for c in cases]},
        {"id":"col","label":"Gender/Number","values":[{"id":c,"label":c} for c in cols]},
    ]
    layout = {"rows":"case", "cols":"col", "pages": None}

    cells = []
    derivs_all = []

    for case in cases:
        for col in cols:
            spec = profile.cell_rule(lex, case, col)
            if not spec:  # пустая клетка
                continue
            if spec["type"] == "form":
                form = phonol(buildForm(lex, spec["gr"]))
                cell_json, derivs = form_to_json(getattr(lex,"name","LEX"),
                                                 grammemes=[x for x in [case, *([tok for tok in col.split('.') if tok != '—'])] if x],
                                                 postfixemes=[],
                                                 form_obj=form,
                                                 cell_label=None)
                for d in derivs:
                    d["lexeme_id"] = lex.name
                    d["gr_obj"]    = spec["gr"] 
                    
                derivs_all += derivs 

                cells.append({
                    "coords": {"case": case, "col": col},
                    "content": {
                        "type": "form",
                        "form_id": cell_json["form_id"],
                        "variants": cell_json["variants"],
                    }
                })

    extra_rows = profile.extra_rows(lex)
    if extra_rows:
        for row in extra_rows:
            if row["kind"] == "form_fullrow":
                form = phonol(buildForm(lex, row["gr"]))
                cell_json, derivs = form_to_json(getattr(lex,"name","LEX"), [], [], form)
                for d in derivs:
                    d["lexeme_id"] = lex.name
                    d["gr_obj"]    = row["gr"]
                derivs_all += derivs
                cells.append({"coords":{"case":"CONSTRUCT_FORM","col":cols[0]},
                              "content":{"type":"form_fullrow","form_id":cell_json["form_id"],"variants":cell_json["variants"],
                                         "label":row.get("label","")}})
                axes[0]["values"].append({"id":"CONSTRUCT_FORM","label":row.get("label","CONSTRUCT form")})
    

            elif row["kind"] == "sublexeme":
                tag = row["tag"]
                rid = f"{tag}"
            
                if not any(v["id"] == rid for v in axes[0]["values"]):
                    axes[0]["values"].append({"id": rid, "label": rid})
            
            
                if "Ag" in lex.Groups and tag == "BAH1" and "DU" in cols:
                    sub_big = sublexeme(lex, ["DU", "BAH1"])
                    label_big = dictionary_form(sub_big, every=True)
                    cells.append({
                        "coords": {"case": "BAH1", "col": "DU"},
                        "content": {
                            "type": "sublexeme",
                            "lexeme_ref": {"sub_tag": "DU.BAH1"},
                            "label": label_big,
                        }
                    })     
                    
                elif lex.name in ["D_kanwe","D_kloz"] and tag == "ADJ" and "DU" in cols:
                    sub_big = sublexeme(lex, ["DU", "ADJ"])
                    label_big = dictionary_form(sub_big, every=True)
                    cells.append({
                        "coords": {"case": "ADJ", "col": "DU"},
                        "content": {
                            "type": "sublexeme",
                            "lexeme_ref": {"sub_tag": "DU.ADJ"},
                            "label": label_big,
                        }
                    })   
                    
                else:
    
                    sub = sublexeme(lex, [tag])
                    label = dictionary_form(sub, every=True)
                    cells.append({
                        "coords": {"case": rid, "col": cols[0]},
                        "content": {
                            "type": "sublexeme",
                            "lexeme_ref": {"sub_tag": tag},
                            "label": label,
                        }
                    })
                
                    if (category == "weak" or "Yetwe" in lex.Groups) and tag == "ADJ" and "PL" in cols:
                        sub_big = sublexeme(lex, ["PL", "ADJ"])
                        label_big = dictionary_form(sub_big, every=True)
                        cells.append({
                            "coords": {"case": "ADJ", "col": "PL"},
                            "content": {
                                "type": "sublexeme",
                                "lexeme_ref": {"sub_tag": "PL.ADJ"},
                                "label": label_big,
                            }
                        })
 
            elif row["kind"] == "big_sublexeme":
                sub = sublexeme(lex, row["tags"])
                label = dictionary_form(sub, every=True)
                rid = ".".join(row["tags"])
                axes[0]["values"].append({"id":rid,"label":rid})
                cells.append({"coords":{"case":rid,"col":cols[0]},
                              "content":{"type":"sublexeme","lexeme_ref":{
                                                                          "sub_tag":rid},
                                         "label":label}})

    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lexeme_id": getattr(lex,"name","LEX"),
        "meaning": lex.Meaning,
        "category": category,
        "axes": axes,
        "layout": layout,
        "cells": cells,
        "_derivations": derivs_all,
    }
    
    
    if save_to:
        import json
        with open(save_to,"w",encoding="utf-8") as f: json.dump(payload,f,ensure_ascii=False,indent=2)
    return payload

DEIXES   = ["DIST","MED","PROX"]
NUMBERS  = ["SG","DU","PL"]
GENDERS  = ["M","F"]


def _build_p3_columns() -> List[str]:
    cols: List[str] = []
    for d in DEIXES:
        cols.append(d)
        cols.append(f"M.SG.{d}")
        cols.append(f"F.SG.{d}")
        cols.append(f"DU.{d}")
        cols.append(f"M.PL.{d}")
        cols.append(f"F.PL.{d}")
    return cols

def _parse_p3_col(tok: str) -> Tuple[str, str | None, str | None]:
    """
    "DIST"          -> ("DIST", None, None)
    "DIST.SG.M"     -> ("DIST", "SG", "M")
    """
    parts = tok.split(".")
    if len(parts) == 1:
        return None, None, parts[0]
    if len (parts) == 2:
        return None, parts[0], parts[1]
    if len(parts) == 3:
        g, n, d = parts
        return g, n, d
    raise ValueError(f"Unexpected P3 column token: {tok}")

def export_paradigm_p3(lex, *, save_to: str | None = None) -> Dict:

    lex_name = getattr(lex, "name", "LEX")

    cols = _build_p3_columns()
    axes = [
        {"id": "case", "label": "Case", "values": [{"id": c, "label": c} for c in CASES_PLUS_ADV]},
        {"id": "col",  "label": "Deixis · Number · Gender", "values": [{"id": c, "label": c} for c in cols]},
    ]
    layout = {"rows": "case", "cols": "col", "pages": None}

    cells: List[Dict] = []
    all_derivations: List[Dict] = []

    for case in CASES_PLUS_ADV:
        for col in cols:
            if case == "ADV" and col not in ["DIST","PROX","MED"]:
                continue
            if case == "ABL3" and col != "DIST":
                continue
            gender, number, deixis = _parse_p3_col(col)

            grams = []
            if gender is not None:
                grams.append(am[gender])
            if number is not None:
                grams.append(am[number])
            grams.extend([am[deixis],am[case]])

            gr = Grammeme(grams)
            form = phonol(buildForm(lex, gr))

            g_for_id = [case, deixis] + ([number] if number else []) + ([gender] if gender else [])

            cell_json, derivs = form_to_json(
                lex_name,
                grammemes=g_for_id,
                postfixemes=[],
                form_obj=form,
                cell_label=None,
            )
            for d in derivs:
                d["lexeme_id"] = lex.name
                d["gr_obj"]    = gr
            all_derivations.extend(derivs)

            cells.append({
                "coords": {"case": case, "col": col},
                "content": {
                    "type": "form",
                    "form_id": cell_json["form_id"],
                    "variants": cell_json["variants"],
                }
            })

    for num_tag in ["SG", "PL"]:
        row_id = f"{num_tag}+OBL"
        axes[0]["values"].append({"id": row_id, "label": row_id})
    
        gr = Grammeme([am[num_tag],am["OBL"]])
        form = phonol(buildForm(lex, gr))
    
        cell_json, derivs = form_to_json(
            lex_name,
            grammemes=["OBL", num_tag],
            postfixemes=[],
            form_obj=form,
            cell_label=None,
        )
        for d in derivs:
            d["lexeme_id"] = lex.name
            d["gr_obj"]    = gr
        all_derivations.extend(derivs)
    
        first_col_id = axes[1]["values"][0]["id"]
    
        cells.append({
            "coords": {"case": row_id, "col": first_col_id},
            "content": {
                "type": "form",
                "form_id": cell_json["form_id"],
                "variants": cell_json["variants"],
            }
        })

    adj_row_id = "ADJ+PL"  # важно: НЕ начинать с "SUBLEXEME."
    axes[0]["values"].append({"id": adj_row_id, "label": "ADJ"})
    
    col_ids = {c["id"] for c in axes[1]["values"]}
    
    for d in DEIXES:        # "DIST","MED","PROX"
        for g in GENDERS:   # "M","F"
            col_id = f"{g}.PL.{d}"
            if col_id not in col_ids:
                continue
    
            sub = sublexeme(lex, [g, "PL", d, "ADJ"])
            label = dictionary_form(sub, every=True)
    
            cells.append({
                "coords": {"case": adj_row_id, "col": col_id},
                "content": {
                    "type": "sublexeme",
                    "lexeme_ref": {"base_id": lex_name, "sub_tag": f"{g}.PL.{d}.ADJ"},
                    "label": label,
                }
            })



    payload =  {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lexeme_id": lex_name,
        "category": getattr(lex, "Category", "") or "P_3",
        "meaning": lex.Meaning,
        "axes": axes,
        "layout": layout,
        "cells": cells,
        "_derivations": all_derivations,
    }

    if save_to:
        import json
        with open(save_to, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload