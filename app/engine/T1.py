from dataclasses import dataclass
from .T_LoadMorphemes import AbstractMorpheme, ContextMorpheme, Letter, am
from .T_LoadLexemes import Lexeme, lx
from .T_LoadRules import rules
from typing import List, Optional, Any
from copy import deepcopy
import re
from pathlib import Path
import json


def _last_base_char(token: str) -> str:
    for ch in reversed(token):
        if ch != "`":
            return ch
    return ""

def _is_vowel_char(ch: str) -> bool:
    return ch in {"a","e","i","o","u","ä","ā","ī","ū"}

def tokenize_shape(shape: Optional[str]) -> List[str]:
    if shape is None:
        return []
    s = str(shape).strip()
    if s == "" or s == "Ø":
        return []

    tokens: List[str] = []
    for ch in s:

        if ch == "h":
            if tokens:
                last_base = _last_base_char(tokens[-1])
                if last_base and not _is_vowel_char(last_base):
                    tokens[-1] = tokens[-1] + "h"
                    continue
            tokens.append("h")
            continue

        if ch in ("i", "u", "ī", "ū"):
            if tokens:
                last_base = _last_base_char(tokens[-1])
                if last_base and _is_vowel_char(last_base):
                    tokens[-1] = tokens[-1] + ch
                    continue
            tokens.append(ch)
            continue

        tokens.append(ch)

    # Post-process: merge pattern ['h', Xh] -> ['hXh'] when X is a consonant token ending with 'h'
    i = 0
    while i < len(tokens) - 1:
        if tokens[i] == "h" and tokens[i + 1].endswith("h"):
            mid_base = _last_base_char(tokens[i + 1])
            if mid_base and not _is_vowel_char(mid_base):
                # merge: 'h' + tokens[i+1] -> one token
                tokens[i] = tokens[i] + tokens[i + 1]
                del tokens[i + 1]
                continue
        i += 1

    return tokens

@dataclass(slots=True)
class Grammeme:
    Morphemes: List["AbstractMorpheme"]

@dataclass(slots=True)
class Postfixeme:
    Morphemes: List["AbstractMorpheme"]

class Variant:
    def __init__(self, m: list["ContextMorpheme"], l: list["Letter"]):
        self.m = m
        self.l = l
     #   self.letter_ids = [lt.id for lt in l]
        self.rules = []
        self.yes_options = []
        self.no_options = []
    def __repr__(self):
        return gloss(self)

@dataclass(slots=True)
class Form:
    Lexeme: "Lexeme"
    Grammeme: "Grammeme"
    Postfixeme: "Postfixeme"
    Variants: List["Variant"]
    def __repr__(self):
        return ", ".join([Variant.__repr__(v) for v in self.Variants])


def buildForm(lex: "Lexeme", gram: "Grammeme", post: "Postfixeme" = None):
    m = []
    l = []
    r = []
    c = -1
    for morpheme in lex.Morphemes:
        new = ContextMorpheme.from_abstract_fast(morpheme)
        m.append(new)
        letters = tokenize_shape(new.shape)
        c += 1
        for let in letters:
            new_letter = Letter(new,let)
            l.append(new_letter)
    
    for cat, idxs in lex.MorphemeCategories.items():
        for idx in idxs:
            if cat == "gender":
                m[idx].Category = "strong"
            elif cat != "verb":
                m[idx].Category = cat
    
    for group, idxs in lex.MorphemeGroups.items():
        for idx in idxs:
            m[idx].Groups.append(group)
    last = m[-1]
    for morpheme in gram.Morphemes:
        new = ContextMorpheme.from_abstract_fast(morpheme)
        if new.Category in ["person","case","voice"]:
            new.wagon = 1
            r.append(f"Grammatical person, case, voice go to the ending: {new.name}")
        elif new.Category == "gender" and last.Category in ["relative","qualitative","numeral"]:
            new.wagon = 1
            r.append(f"Gender goes to the ending after relatives, qualitatives and numerals {new.name}")
        elif new.Category == "number":
            if last.Category == "double":
                if new.name == "PL":
                    new.wagon = 1
                    r.append("PL goes to the ending after doubles")
            elif last.Category not in ["weak","person"] and not (len(m) > 1 and m[-1].Category == "gender" and m[-2].Category == "person") \
                and not (last.Category == "numeral" and last.name != "ṣomă"):
                c += 1
                new.wagon = 1
                r.append(f"Number goes to the ending except after weaks, doubles, persons, persons+genders and numerals except ṣomă: {new.name}") # person+gender for nṣac (**nṣanac) and P_3
        else:
            if morpheme.Category != "gender":
                last = morpheme
            r.append(f"Grammatical morphemes are attached to the stem: {new.name}")
        m.append(new)
        letters = tokenize_shape(new.shape)
        for let in letters:
            new_letter = Letter(new,let)
            l.append(new_letter)
    if post:
        for morpheme in post.Morphemes:
            new = ContextMorpheme.from_abstract_fast(morpheme)
            m.append(new)    
            new.wagon = 0
            letters = tokenize_shape(new.shape)
            for let in letters:
                new_letter = Letter(new,let)
                l.append(new_letter)
    variants = [Variant(m, l)]
    gl = "-".join([mor.meaning for mor in m])
    gl = re.sub(" ",".",gl)
    gl2 = "-".join([mor.shape for mor in m])
    variants[0].rules.append(("0","","",gl))
    variants[0].rules.append(("1","","",gl2))
    return Form(lex,gram,post,variants)

def is_root(m):
    return m.name[0].islower()


def before_letter(l: List["Letter"], i: int):
    before = ""
    j = 0
    while j != i:
        before += l[j].shape
        j += 1
    return before

def after_letter(l: List["Letter"], i: int):
    if i == len(l)-1:
        return ""
    after = ""
    for j in range(i+1,len(l)):
        after += l[j].shape
    return after

def satisfy_phon(variant: "Variant", i: int, condition: Any, *, unless: bool = False) -> bool:
    if condition.startswith("*"):
        subcondition = condition[1:].split("—")
        return (all if unless else any)(satisfy_phon(variant, i, cond) for cond in subcondition)
    l = variant.l
    by, what, of = condition.split("~")
    by = int(by)
    of = of.split(",")
    j = i + by         
    if j < 0 or j > len(l)-1:
        return False
    if what == "Shape":
        return any(l[j].shape == x for x in of)
    elif what == "wagon":
        return l[j].morpheme.wagon == int(of[0])
    elif what == "Before":
        pattern = of[0].format(**replacements)
        if not re.search(pattern,before_letter(l, j)):
            return False
    elif what == "After":
        pattern = of[0].format(**replacements)
        if not re.search(pattern,after_letter(l, j)):
            return False
    elif what == "Morpheme" and l[j].morpheme not in of:
        return False
    elif what == "Category" and l[j].morpheme.Category not in of:
        return False
    elif what == "Color" and l[j].morpheme.Color not in of:
        return False
    elif what == "Class":
        pattern = of[0].format(**replacements)
        return l[j].shape in pattern
    elif what == "Group" and not any(x in of for x in l[j].morpheme.Groups):
        return False
    elif what == "Native" and l[j].morpheme.Language:
        return False
    elif what == "Foreign" and not l[j].morpheme.Language:
        return True
    elif what == "Bhe":
        if j == len(l)-1:
            return True
        if l[j+1].morpheme == "EMPH":
            return True
        return False
    elif what == "Bho":
        if j == len(l)-1:
            return True
        if l[j+1].shape == "`" and j == len(l)-2: # for psāri
            return True
        if l[j+1].shape in _C:
            return True
        if l[j+1].morpheme == "EMPH":
            return True
        return False
    elif what == "Bha":
        if j == len(l)-1:
            return True
        if l[j].morpheme == l[j+1].morpheme:
            return False
        if ((l[j+1].morpheme in ["INS","GEN","ADV","PLUS","ADJ","PL","DU"] or l[j+1].morpheme.wagon == 0) and re.search(r"^[ynṣpt]",l[j+1].morpheme.shape)) or l[j+1].morpheme == "EMPH":
            return True
        return False
    elif what == "Root":
        return is_root(l[j].morpheme)
    elif what == "AfterAnother":
        return j > 0 and l[j].shape != l[j-1].shape
    elif what == "AfterAnotherMorpheme":
        return j > 0 and l[j].morpheme != l[j-1].morpheme
    elif what == "BeforeAnotherMorpheme":
        return j < len(l)-1 and l[j].morpheme != l[j+1].morpheme
    elif what == "BetweenDifferent":
        return len(l)-1 > j > 0 and l[j-1].shape != l[j+1].shape
    elif what == "BeforeSimilar":
        return len(l)-2 > j and l[j+1].shape == l[j+2].shape and l[j+1].shape in _C
    elif what == "PermissibleCluster":
        cluster = ""
        if j > 0:
            for k in range(j-1,-1,-1):
                if l[k].shape in _vowel:
                    break
                cluster = l[k].shape + cluster
        if j < len(l)-1:
            for k in range(j+1,len(l)):
                if l[k].shape in _vowel:
                    break
                cluster  += l[k].shape
        if len(cluster) < 2:
            return True
        if cluster[-1] == cluster[-2]:
            cluster = cluster[:-1]
        if len(cluster) == 2:
            return True
        if cluster in ["rmnt","mncs","mptr","ṣln","ynty"]:
            return True
        subclaster = cluster[1:-1]
        if re.search(r"[rlλy]",subclaster):
            return False
        return len(cluster) < 4
    elif what == "PermissibleClusterLast":
        cluster = ""
        if j > 0:
            for k in range(j-1,-1,-1):
                if l[k].shape in _vowel:
                    break
                cluster = l[k].shape + cluster
        if j < len(l)-1:
            for k in range(j+1,len(l)):
                if l[k].shape in _vowel:
                    break
                cluster  += l[k].shape
        return cluster in ["rs"]
    elif what == "Bhu":
        if l[j].shape not in ["ä","a","ā","ă"]:
            return False
        if j < len(l)-1:
            if l[j+1].morpheme.id == l[j].morpheme.id:
                return False
            if l[j+1].morpheme.Category == "case" and l[j+1].morpheme.wagon == 1:
                return True
            if l[j+1].morpheme in ["EMPH","PLUS"]:
                return True
        curmorid = l[j].morpheme.id
        k = 0
        while variant.m[k].id != curmorid:
            k += 1
        if k == len(variant.m)-1:
            return False
        if not any(mor.wagon == 1 for mor in variant.m[k:]):
            return False
        if j == len(l)-1:
            return True
        return False
    elif what == "Bhi":
        if j < len(l)-1:
            if l[j+1].morpheme.id == l[j].morpheme.id:
                return False
            if l[j+1].morpheme.name.startswith(("QUAL","ANIM","ADJ")):
                return True
        curmorid = l[j].morpheme.id
        k = 0
        while variant.m[k].id != curmorid:
            k += 1
        if k == len(variant.m)-1:
            return True
        return False
    return True

def satisfy_morph(variant: "Variant", i: int, condition: Any, *, unless: bool = False) -> bool:
    if condition.startswith("*"):
        subcondition = condition[1:].split("—")
        return (all if unless else any)(satisfy_morph(variant, i, cond) for cond in subcondition)
    m = variant.m
    if len(condition.split("~")) != 3:
        print(condition)
    by, what, of = condition.split("~")
    by = int(by)
    of = of.split(",")
    j = i + by
    if j < 0 or j > len(m)-1:
        return False
    elif what == "Category" and m[j].Category not in of:
        return False
    elif what == "Morpheme" and m[j] not in of:
        return False
    elif what == "Color" and m[j].Color not in of:
        return False
    elif what == "Group" and not any(x in of for x in m[j].Groups):
        return False
    elif what == "Short":
        return(short(m[j].shape))
    elif what == "Pattern":
        pattern = of[0].format(**replacements)
        if not re.search(pattern,m[j].shape):
            return False
    elif what == "Native" and m[j].Language:
        return False
    elif what == "Causativity":
        return m[j].Causativity and of[0] in m[j].Causativity
    elif what == "Voice":
        return m[j].Voice and of[0] in m[j].Voice
    elif what == "Root":
        return is_root(m[j])
    elif what == "Shape":
        return any(m[j].shape == x for x in of)
    elif what == "Empty":
        return m[j].shape == "Ø" or m[j].shape == ""
    elif what == "NonEmpty":
        return m[j].shape != "Ø" and m[j].shape != ""
    elif what == "wagon":
        return m[j].wagon == int(of[0])
    elif what == "Gender":
        if m[j].Gender:
            return of[0] == m[j].Gender
        else:
            return False
    elif what == "Before":
        pattern = of[0].format(**replacements)
        if not re.search(pattern,before_morph(variant, m[j])):
            return False
    elif what == "After":
        if of[0] == "Ø":
            return after_morph(variant, m[j]) == ""
        else:
            pattern = of[0].format(**replacements)
        if not re.search(pattern,after_morph(variant, m[j])):
            return False
    elif what == "Last":
        return j == len(m)-1
    return True

def gloss(v: "Variant") -> str:
    m = v.m
    l = v.l
    ans = ""
    for k in range(len(m)):
        morph_letters = [lt for lt in l if getattr(lt, "morpheme", None) is m[k]]
        if not morph_letters:
            morph_part = "Ø"
        else:
            morph_part = "".join(lt.shape for lt in morph_letters)
        ans += morph_part
        if k < len(m) - 1:
            ans += "-"
    return ans

def before_morph(v: "Variant", mor: "ContextMorpheme") -> str:
    l = v.l
    m = v.m
    x = mor.id

    y = 0
    while m[y].id != x:
        y += 1

    while y >= 0 and (m[y].id == x or m[y].shape == "" or m[y].shape == "Ø"):
        y -= 1

    n = len(l)
    i = n - 1
    empty = False
    while i >= 0 and l[i].morpheme.id != x:
        i -= 1
    if i == -1:
        empty = True

    before = ""
    if empty:
        if y == -1:
            return before
        z = n - 1
        while z >= 0 and l[z].morpheme.id != m[y].id:
            z -= 1
        if z < 0:
            return before
        parts = []
        k = 0
        while k <= z:
            parts.append(l[k].shape)
            k += 1
        return "".join(parts)
    else:
        j = 0
        while j < n and l[j].morpheme.id != x:
            j += 1
        parts = []
        k = 0
        while k < j:
            parts.append(l[k].shape)
            k += 1
        return "".join(parts)
     
def after_morph(v: "Variant", mor: "ContextMorpheme") -> str:
    l = v.l
    m = v.m
    x = mor.id
    y = 0
    while m[y].id != x:
        y += 1
    while y < len(m) and (m[y].id == x or m[y].shape == "" or m[y].shape == "Ø"):
        y += 1
    n = len(l)
    i = 0
    empty = False
    while i < n and l[i].morpheme.id != x:
        i += 1
    if i == n:
        empty = True
    after = ""
    if empty:
        if y == len(m):
            return after
        z = 0
        while z < n and l[z].morpheme.id != m[y].id:
            z += 1
        while z < n:
            after += l[z].shape
            z += 1
    else:
        while i < n and l[i].morpheme.id == x:
            i += 1
        while i < n:
            after += l[i].shape
            i += 1
    return after

def replace_letters(v: "Variant", mor: "ContextMorpheme", new_letters: str) -> List["Letter"]:
    l = v.l
    m = v.m
    x = mor.id
    y = 0
    while m[y].id != x:
        y += 1
    while y < len(m) and (m[y].id == x or m[y].shape == "" or m[y].shape == "Ø"):
        y += 1
    n = len(l)
    i = 0
    empty = False
    while i < n and l[i].morpheme.id != x:
        i += 1
    if i == n:
        empty = True
    
    if empty:
        if y == len(m):
            i = j = len(l)
        else:
            z = 0
            while z < n and l[z].morpheme.id != m[y].id:
                z += 1
            i = j = z
    else:
        j = i
        while j < n and l[j].morpheme.id == x:
            j += 1

    tok = tokenize_shape(new_letters)
    news = [Letter(mor, t) for t in tok]
    l[i:j] = news
    return l

_long = 'iuaāeoīūṛ'
_vowel = 'äăiuaāeoīūṛ'
_Y = 'yw'
_N = 'nmñṅṇ'
_C = 'kctzpgdbqṭḍjhśsṣnmñṅṇλlrywvf'
_T = 'kctzpgdbqṭḍjf'
_L = 'lr'
_S = 'śṣs'


replacements = {
    '_long': 'iuaāeoīūṛ',
    '_V': 'äăiuaāeoīūṛ',
    '_L': 'lr',
    '_Q': 'kctzpgdbqṭḍjhśsṣf',
    '_C': 'kctzpgdbqṭḍjhśsṣnmñṅṇλlrywvf',
    '_Y': 'yw',
    '_N': 'nmñṅṇ',
    '_T': 'kctzpgdbqṭḍj',
    '_S': 'śṣs'
}

def apply(form: "Form", r: str) -> "Form":
    if not form.Variants or r.startswith("IX."):
        return form
    rule = rules[r]

    def log_rule(var, rule_id, position, optional_flag):
        if not hasattr(var, "rules") or var.rules is None:
            var.rules = []
        var.rules.append((rule_id, position, optional_flag, gloss(var)))

    def check_phon_conditions(subvar, pos):
        ok = all(satisfy_phon(subvar, pos, cond) for cond in rule.ifs)
        skip = any(satisfy_phon(subvar, pos, cond, unless=True) for cond in rule.unlesses)
        return ok and not skip

    def will_change_phon(subvar, pos):
        for op in rule.operations:
            parts = op.split("~")
            if len(parts) != 3:
                continue
            by_str, what, of = parts
            try:
                by = int(by_str)
            except ValueError:
                continue
            j = pos + by
            if j < 0 or j >= len(subvar.l):
                continue
            if what in ("Prefix", "Gemination", "Desanskr", "Palat", "Suf") or (what == "Shape" and (of == 'Ø' or tokenize_shape(of))):
                return True
            elif what == "Assim" and j + 1 < len(subvar.l) and subvar.l[j].shape != subvar.l[j + 1].shape:
                return True
        return False

    def apply_phon_operations(var, pos):
        changed = False
        desanskr_map = {"kh": "k","g": "k","gh": "k","ch": "c","j": "c","jh": "c","ṭ": "t", "ṭh": "t","ḍ": "t","ḍh": "t","ṇ": "n","th": "t", "d": "t", "dh": "t","ph": "p", "b": "p", "bh": "p","v": "w"}
        palat_map = {'k': 'ś', 'q': 'ś', 'z': 'ś', 'ṣ': 'ś','l': 'λ','n': 'ñ', 'ṅ': 'ñ', 'ṇ': 'ñ','t': 'c','s': 'ṣ'}
        keys_d = sorted(desanskr_map, key=len, reverse=True)
        keys_p = sorted(palat_map, key=len, reverse=True)
        pattern_d = re.compile("|".join(re.escape(k) for k in keys_d)) if keys_d else None
        pattern_p = re.compile("|".join(re.escape(k) for k in keys_p)) if keys_p else None
        for op in rule.operations:
            parts = op.split("~")
            if len(parts) != 3:
                continue
            by_str, what, of = parts
            try:
                by = int(by_str)
            except ValueError:
                continue
            j = pos + by
            if j < 0 or j >= len(var.l):
                continue
            if what == "Prefix":
                tokens = tokenize_shape(of)
                if not tokens:
                    continue
                template = var.l[j]
                target = getattr(template, "morpheme", None)
                old_len = len(var.l)
                for tok in reversed(tokens):
                    new_l = deepcopy(template)
                    new_l.shape = tok
                    new_l.morpheme = target
                    var.l.insert(j, new_l)
                if target:
                    target.shape = "".join(lt.shape for lt in var.l if getattr(lt, "morpheme", None) is target)
                if len(var.l) != old_len:
                    changed = True
                    
            elif what == "Shape":
                template = var.l[j]
                target = getattr(template, "morpheme", None)
                old_shape = template.shape
                old_len = len(var.l)
                if of == 'Ø':
                    del var.l[j]
                    changed = True
                else:
                    tokens = tokenize_shape(of) or []
                    del var.l[j]
                    for off, tok in enumerate(tokens):
                        new_l = deepcopy(template)
                        new_l.shape = tok
                        new_l.morpheme = target
                        var.l.insert(j + off, new_l)
                    new_str = "".join(tokens)
                    if new_str != old_shape:
                        changed = True
                if target:
                    target.shape = "".join(lt.shape for lt in var.l if getattr(lt, "morpheme", None) is target)

            elif what == "Gemination":
                template = var.l[j]
                old_len = len(var.l)
                new_l = deepcopy(template)
                new_l.morpheme = getattr(template, "morpheme", None)
                var.l.insert(j + 1, new_l)
                target = getattr(template, "morpheme", None)
                if target:
                    target.shape = "".join(lt.shape for lt in var.l if getattr(lt, "morpheme", None) is target)
                if len(var.l) != old_len:
                    changed = True

            elif what == "Suf":
                template = var.l[j]
                target = getattr(template, "morpheme", None)
                if of == 'Ø':
                    pass
                else:
                    tokens = tokenize_shape(of) or []
                    if tokens:
                        for off, tok in enumerate(tokens, start=1):
                            new_l = deepcopy(template)
                            new_l.shape = tok
                            new_l.morpheme = target
                            var.l.insert(j + off, new_l)
                        changed = True
                if target:
                    target.shape = "".join(
                        lt.shape for lt in var.l if getattr(lt, "morpheme", None) is target
                    )

            elif what == "Desanskr" and pattern_d:
                letter = var.l[j]
                orig = letter.shape
                new_shape = pattern_d.sub(lambda m: desanskr_map[m.group(0)], orig)
                if new_shape != orig:
                    letter.shape = new_shape
                    changed = True
                    target = getattr(letter, "morpheme", None)
                    if target:
                        target.shape = "".join(lt.shape for lt in var.l if getattr(lt, "morpheme", None) is target)
            elif what == "Palat" and pattern_p:
                letter = var.l[j]
                orig = letter.shape
                new_shape = pattern_p.sub(lambda m: palat_map[m.group(0)], orig)
                if new_shape != orig:
                    letter.shape = new_shape
                    changed = True
                    target = getattr(letter, "morpheme", None)
                    if target:
                        target.shape = "".join(lt.shape for lt in var.l if getattr(lt, "morpheme", None) is target)
            elif what == "Assim":
                if j + 1 >= len(var.l):
                    continue
                letter = var.l[j]
                next_shape = var.l[j + 1].shape
                orig = letter.shape
                if next_shape != orig:
                    letter.shape = next_shape
                    changed = True
                    target = getattr(letter, "morpheme", None)
                    if target:
                        target.shape = "".join(lt.shape for lt in var.l if getattr(lt, "morpheme", None) is target)
        return changed

    def process_phon(start_len, reverse=False):
        step = -1 if reverse else 1
        for idx in range(start_len):
            var = form.Variants[idx]
            i = len(var.l) - 1 if reverse else 0
            active = [var]
            while (reverse and i >= 0) or (not reverse and i < len(active[0].l)):
                new_active = []
                for subvar in active:
                    if (reverse and i >= len(subvar.l)) or (not reverse and i < 0):
                        new_active.append(subvar)
                        continue
                    if not check_phon_conditions(subvar, i) or not will_change_phon(subvar, i):
                        new_active.append(subvar)
                        continue
                    optional = getattr(rule, "optional", False)
                    if optional:
                        opt_str = optional[0]
                    else:
                        opt_str = ""
                    changed = False
                    if optional:
                        no_var = deepcopy(subvar)
                        no_var.no_options.append((r,opt_str))
                 #       log_rule(no_var, r, i, "")
                        subvar.yes_options.append((r,opt_str))
                        changed = apply_phon_operations(subvar, i)
                        if changed:
                            log_rule(subvar, r, i, opt_str)
                        new_active += [subvar, no_var]
                    else:
                        changed = apply_phon_operations(subvar, i)
                        if changed:
                            log_rule(subvar, r, i, "")
                        new_active.append(subvar)
                active = new_active
                i += step
            for sv in active:
                if sv is not var:
                    form.Variants.append(sv)

    def check_morph_conditions(var, pos):
        ok = all(satisfy_morph(var, pos, cond) for cond in rule.ifs)
        skip = any(satisfy_morph(var, pos, cond, unless=True) for cond in rule.unlesses)
        return ok and not skip

    def will_change_morph(var, pos):
        for op in rule.operations:
            parts = op.split("~")
            if len(parts) != 3:
                continue
            by_str, what, of = parts
            try:
                by = int(by_str)
            except ValueError:
                continue
            j = pos + by
            if j < 0 or j >= len(var.m):
                continue
            if what in ("del", "Suffix", "Prefix", "Desanskr") or \
               (what == "Color" and getattr(var.m[j], "Color", None) != of) or \
               (what == "wagon" and getattr(var.m[j], "wagon", None) != int(of)) or \
               (what == "Voice" and getattr(var.m[j], "Voice", None) != of) or \
               (what == "Shape" and (of == 'Ø' or getattr(var.m[j], "shape", None) != of)) or \
               (what == "Group" and of not in getattr(var.m[j], "Groups", [])) or \
               (what == "Change" and re.search(of.split("→")[0].format(**replacements), var.m[j].shape)) or \
               (what == "PalatInit" and len(var.m[j].shape) > 1 and var.m[j].shape[1] != '`') or \
               (what == "Ablaut" and last_vowel_index(var.m[j].shape) and var.m[j].shape[last_vowel_index(var.m[j].shape)] in 'äā') or \
               (what == "VocalRedup" and first_vowel_index(var.m[(j-1 if var.m[j].Category == "causativity" else j)].shape) is not None) or \
               (what == "ConsRedup" and first_vowel_index(var.m[j].shape) and not re.search(f"^{_Y}{_long}{_N}", var.m[j].shape)) or \
               what in ("AblautGenderM", "AblautGenderF"):
                return True
        return False

    def apply_morph_operations(var, pos, idx):
        changed = False
        desanskr_map = {"kh": "k","g": "k","gh": "k","ch": "c","j": "c","jh": "c","ṭ": "t", "ṭh": "t","ḍ": "t","ḍh": "t","ṇ": "n","th": "t", "d": "t", "dh": "t","ph": "p", "b": "p", "bh": "p","v": "w"}
        keys_d = sorted(desanskr_map, key=len, reverse=True)
        pattern_d = re.compile("|".join(re.escape(k) for k in keys_d)) if keys_d else None
        old_shapes = {m.id: m.shape for m in var.m if hasattr(m, 'id')}
        for op in rule.operations:
            parts = op.split("~")
            if len(parts) != 3:
                continue
            by_str, what, of = parts
            try:
                by = int(by_str)
            except ValueError:
                continue
            j = pos + by
            if j < 0 or j >= len(var.m):
                continue
            mor = var.m[j]
            old_shape = mor.shape
            if what == "Color":
                if getattr(mor, "Color", None) != of:
                    setattr(mor, "Color", of)
                    changed = True
            if what == "wagon":
                if getattr(mor, "wagon", None) != int(of):
                    setattr(mor, "wagon", int(of))
                    changed = True
            elif what == "Voice":
                if getattr(mor, "Voice", None) != of:
                    setattr(mor, "Voice", of)
                    changed = True
            elif what == "del":
                indices_to_delete.append(idx)
                changed = True
            elif what == "Shape":
                of2 = '' if of == 'Ø' else of
                if getattr(mor, "shape", None) != of2:
                    setattr(mor, "shape", of2)
                    replace_letters(var, mor, of2)
                    changed = True
            elif what == "Group":
                if of not in getattr(mor, "Groups", []):
                    mor.Groups.append(of)
                    if of == "PrsEnd" and "PrtEnd" in mor.Groups:
                        mor.Groups.remove("PrtEnd")
                    elif of == "PrtEnd" and "PrsEnd" in mor.Groups:
                        mor.Groups.remove("PrsEnd")
                    changed = True
            elif what == "Change":
                before, after = of.split("→")
                after = '' if after == 'Ø' else after
                before_fmt = before.format(**replacements)
                if re.search(before_fmt, mor.shape):
                    mor.shape = re.sub(before_fmt, after, mor.shape)
                    replace_letters(var, mor, mor.shape)
                    changed = True
            elif what == "Suffix":
                mor.shape += of
                replace_letters(var, mor, mor.shape)
                changed = True
            elif what == "PalatInit":
                if len(mor.shape) > 1 and mor.shape[1] != '`':
                    mor.shape = mor.shape[0] + "`" + mor.shape[1:]
                    replace_letters(var, mor, mor.shape)
                    changed = True
            elif what == "Prefix":
                mor.shape = of + mor.shape
                replace_letters(var, mor, mor.shape)
                changed = True
            elif what == "Ablaut":
                idx_v = last_vowel_index(mor.shape)
                if idx_v:
                    mor.shape = mor.shape[:idx_v] + of + mor.shape[idx_v+1:]
                    replace_letters(var, mor, mor.shape)
                    changed = True
            elif what == "VocalRedup":
                j2 = j - 1 if var.m[j].Category == "causativity" else j
                idx_v = first_vowel_index(var.m[j2].shape)
                if idx_v is not None:
                    redup = "ā" if var.m[j2].shape[idx_v] not in "äă" else "a"
                    if var.m[j2] == "bärk":
                        redup = "o"
                    if var.m[j2] in ["läṅk","kon","ag","salu","ālăk"]:
                        redup = "y"
                    if idx_v == 0:
                        redup += "n"
                    old_j2_shape = var.m[j2].shape
                    var.m[j2].shape = redup + var.m[j2].shape
                    replace_letters(var, var.m[j2], var.m[j2].shape)
                    if var.m[j2].shape != old_j2_shape:
                        changed = True
            elif what == "ConsRedup":
                idx_v = first_vowel_index(mor.shape)
                if idx_v and not re.search(f"^[{_Y}][{_long}][{_N}]", mor.shape):
                    redup_v = "ā" if mor.shape[idx_v] not in "äă" else "a"
                    redup = mor.shape[:idx_v][0] + re.sub(rf"[{_C}]", '', mor.shape[:idx_v][1:]) + redup_v
                    old_shape = mor.shape
                    mor.shape = redup + mor.shape
                    mor.Groups.append("Reduplicated")
                    replace_letters(var, mor, mor.shape)
                    if mor.shape != old_shape:
                        changed = True
            elif what == "AblautGenderM":
                x = var.m[j].id
                y = 0
                while y < len(var.m) and var.m[y].id != x:
                    y += 1
                while y < len(var.m) and var.m[y].id == x:
                    y += 1
                old_shape_j = var.m[j].shape
                old_shape_y = var.m[y].shape if y < len(var.m) else None
                var.m[j].shape = var.m[j].shape[:-2] + var.m[y].shape + var.m[j].shape[-1]
                replace_letters(var, var.m[j], var.m[j].shape)
                var.m[y].shape = ""
                replace_letters(var, var.m[y], var.m[y].shape)
                if var.m[j].shape != old_shape_j or (old_shape_y != "" if old_shape_y is not None else False):
                    changed = True
            elif what == "AblautGenderF":
                x = var.m[j].id
                y = 0
                while y < len(var.m) and var.m[y].id != x:
                    y += 1
                while y < len(var.m) and var.m[y].id == x:
                    y += 1
                old_shape_j = var.m[j].shape
                old_shape_y = var.m[y].shape if y < len(var.m) else None
                var.m[j].shape = var.m[j].shape[:-2] + var.m[y].shape + var.m[j].shape[-1]
                if var.m[j+3] == "NOM":
                    replace_letters(var, var.m[j], var.m[j].shape)
                    oldvar2 = deepcopy(var)
                    oldvar2.no_options.append(("F optionally → Ø after gender-ablaut before SG+NOM","+"))
                    form.Variants.append(oldvar2)
                    var.yes_options.append(("F optionally → Ø after gender-ablaut before SG+NOM","+"))
                    var.m[y].shape = ""
                replace_letters(var, var.m[y], var.m[y].shape)
                if var.m[j].shape != old_shape_j or (old_shape_y != var.m[y].shape if old_shape_y is not None else False):
                    changed = True
            elif what == "Desanskr" and pattern_d:
                orig = mor.shape or ""
                new_shape = pattern_d.sub(lambda m: desanskr_map[m.group(0)], orig)
                if new_shape != orig:
                    mor.shape = new_shape
                    try:
                        replace_letters(var, mor, new_shape)
                    except:
                        # fallback omitted for brevity, add if needed
                        pass
                    changed = True
        # Final check if any shape changed
        if not changed:
            current_shapes = {m.id: m.shape for m in var.m if hasattr(m, 'id')}
            changed = current_shapes != old_shapes
        return changed

    if rule.kind == 'phonological':
        process_phon(len(form.Variants), reverse=True)
    elif rule.kind == 'phonological_right':
        process_phon(len(form.Variants), reverse=False)
    elif rule.kind == 'morphological':
        indices_to_delete = []
        num_m = len(form.Variants[0].m) if form.Variants else 0
        for pos in range(num_m):
            start_len = len(form.Variants)
            for idx in range(start_len):
                var = form.Variants[idx]
                if not check_morph_conditions(var, pos) or not will_change_morph(var, pos):
                    continue
                optional = getattr(rule, "optional", False)
                if optional:
                    opt_str = optional[0]
                else:
                    opt_str = ""
                changed = False
                if optional:
                    oldvar = deepcopy(var)
                    oldvar.no_options.append((r,opt_str))
                 #   log_rule(oldvar, r, pos, "")
                    form.Variants.append(oldvar)
                    var.yes_options.append((r,opt_str))
                    changed = apply_morph_operations(var, pos, idx)
                    if changed:
                        log_rule(var, r, pos, opt_str)
                else:
                    changed = apply_morph_operations(var, pos, idx)
                    if changed:
                        log_rule(var, r, pos, opt_str)
        indices_to_delete = list(set(indices_to_delete))
        indices_to_delete.sort(reverse=True)
        for j in indices_to_delete:
            if 0 <= j < len(form.Variants):
                del form.Variants[j]
    return form
    

def standard(v: "Variant") -> bool:
    return not any(x[1] in ["M","V","Post"] for x in v.yes_options) and not any(x[1] in ["P","Pr"] for x in v.no_options)

def showvar(v: "Variant") -> str:
    ans = ""
    for i, let in enumerate(v.l):
        if let.morpheme.wagon == 0 and v.l[i-1].morpheme.wagon != 0:
            ans += "="
        ans += let.shape
    return ans


def phonol(form: "Form") -> "Form":
    for r in rules:
        apply(form, r)
    seen = []
    idxs = []
    indices_to_delete = []
    for i, v in enumerate(form.Variants):
        f = showvar(v)
        if f in seen:
            j = idxs[seen.index(f)]
            if not standard(form.Variants[j]) and standard(form.Variants[i]):
                indices_to_delete.append(j)
            elif not standard(form.Variants[i]) and standard(form.Variants[j]):
                indices_to_delete.append(i)
                continue
            # THIS IS NEW! For kinds of letaṣ: less optional prevail
            elif len(form.Variants[j].yes_options) > len(form.Variants[i].yes_options):
                indices_to_delete.append(j)
            elif len(form.Variants[i].yes_options) > len(form.Variants[j].yes_options):
                indices_to_delete.append(i)
                continue
            elif len(form.Variants[j].rules) > len(form.Variants[i].rules):
                indices_to_delete.append(j)
            else:
                indices_to_delete.append(i)
                continue
        idxs.append(i)
        seen.append(f)
    indices_to_delete.sort(reverse=True)
    for j in indices_to_delete:
        if 0 <= j < len(form.Variants):
            del form.Variants[j]
    return form
        



def short(s: str):
    if 'ai' in s or 'au' in s:
        return False
    if any(long in s for long in _long):
        return False
    return True

def last_vowel_index(s:str):
    for index in range(len(s) - 1, -1, -1):
        if s[index] in _vowel:
            return index

def first_vowel_index(s:str):
    for index in range(len(s)):
        if s[index] in _vowel:
            return index

def show_form(form: "Form", *, st = True):
    if st:
        return ["".join([let.shape for let in v.l]) for v in form.Variants if standard(v)]
    else:
        return ["".join([let.shape for let in v.l]) for v in form.Variants]

def get_paradigm(lexeme: str):
    lex = lx[lexeme]
    cat = lex.CategoryMorphemeCategories
    paradigm = {}
    if cat == "indeclinable":
        gr = Grammeme([])
        form = phonol(buildForm(lex,gr))
        paradigm["—"] = form
    return paradigm

def _append_last_index_to_maps(lex: "Lexeme"):
    if not isinstance(getattr(lex, "MorphemeCategories", None), dict):
        lex.MorphemeCategories = {}
    if not isinstance(getattr(lex, "MorphemeGroups", None), dict):
        lex.MorphemeGroups = {}

    if not getattr(lex, "Morphemes", None):
        lex.MorphemeCategories.clear()
        lex.MorphemeGroups.clear()
        return

    last_idx = len(lex.Morphemes) - 1

    cat = getattr(lex, "Category", None)
    if cat:
        key = str(cat)
        lst = lex.MorphemeCategories.setdefault(key, [])
        if not lst or lst[-1] != last_idx:
            lst.append(last_idx)

    for g in (getattr(lex, "Groups", []) or []):
        key = str(g)
        lst = lex.MorphemeGroups.setdefault(key, [])
        if not lst or lst[-1] != last_idx:
            lst.append(last_idx)

def sublexeme(lex: "Lexeme", mors: list[str], *, am=am) -> "Lexeme":
    if not mors:
        return lex.clone()

    added = []
    for mor in mors:
        mobj = am.get(mor)
        if mobj is None:
            raise KeyError(f"Unknown morpheme '{mor}' for sublexeme")
        added.append(mobj)

    new = lex.clone()
    new.Morphemes.extend(added)

    base_name = getattr(lex, "name", "LEX")
    base_mean = getattr(lex, "Meaning", "") or ""
    new.name = base_name + "".join(f"+{getattr(m,'name',str(m))}" for m in added)
    new.Meaning = base_mean + "".join(
        f"+{getattr(m,'meaning')}" for m in added if getattr(m, "meaning", None)
    )

    last_cat = getattr(added[-1], "Category", None)
    if last_cat:
        if str(last_cat).lower() == "gender":
            new.Category = "strong"
        elif last_cat and str(last_cat).lower() != "verb":
            new.Category = last_cat
    new.Groups = added[-1].Groups

    _append_last_index_to_maps(new)
    return new


def dictionary_form(lex: Lexeme, *, every: bool=False) -> str:
    if lex.Category == "verb":
        g = Grammeme([am["GER2"],am["NMZ"],am['SG'],am['NOM']])
    elif lex.name == "P_3":
        g = Grammeme([am["DIST"],am['NOM']])
    elif lex.name == "P_änzaṃ":
        g = Grammeme([am["M"],am["SG"],am["MED"],am['NOM']])
    elif lex.name in ["N_ṣomă","P_1"] or lex.Category in ["qualitative","relative"]:
        if lex.name == "R_arkănt":
            return "arkant-"
        g = Grammeme([am["M"],am['SG'],am['NOM']])
    elif lex.name in ["N_vä","N_täräy"]:
        g = Grammeme([am["M"],am['NOM']])
    elif lex.Category in ["name","creature","strong","weak","double","person"]:
        g = Grammeme([am["SG"],am['NOM']])
    else:
        g = Grammeme([am["NOM"]])
    form = buildForm(lex,g)
    phonol(form)
    reg = show_form(form)
    if reg:
        if every:
            return ", ".join([r for r in reg])
        if "kāsu" in reg:
            return "kāsu"
        return reg[-1]
    else:
        return "—"

def add_dictionary_form_to_json(json_path: str, output_path: str | None = None):
    jpath = Path(json_path)
    if not jpath.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(jpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated = 0

    for lex_id, raw in data.items():
        if raw["DictionaryForm"]:
            break
        lex = lx[lex_id]

        try:
            df = dictionary_form(lex)
        except Exception as e:
            print(f"WARNING: dictionary_form failed for {lex_id}: {e}")
            df = ""

        raw["DictionaryForm"] = df
        updated += 1

    print(f"Updated {updated} lexemes")

    out = Path(output_path) if output_path else jpath
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved updated JSON to: {out}")

add_dictionary_form_to_json("./data/lexemes.json")

# lex = lx["V_läyt"]
# gr = Grammeme([am["CON"],am["3"],am["SG"],am["ACT"]])
# print(lex)
# form = phonol(buildForm(lex, gr))
# print(form.Variants[0].yes_options)
