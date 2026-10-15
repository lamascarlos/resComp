#!/usr/bin/env python3
"""Convert Unicode / ASCII math in markdown prose to LaTeX ($...$): Greek letters, relations, powers, scientific
notation, subscripted variables (V_set -> $V_{\\mathrm{set}}$). Skips fenced code, $$ blocks, inline code, existing
$...$ spans and link targets. Idempotent. Hand-check equations afterwards.

    python scripts/md_math_to_latex.py FILE.md ...          # dry run (diff)
    WRITE=1 python scripts/md_math_to_latex.py FILE.md ...  # rewrite in place
"""
import os
import re
import sys

GREEK = {"α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\varepsilon", "η": r"\eta",
         "θ": r"\theta", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu", "ν": r"\nu", "ξ": r"\xi", "π": r"\pi",
         "ρ": r"\rho", "σ": r"\sigma", "τ": r"\tau", "φ": r"\phi", "χ": r"\chi", "ω": r"\omega", "Δ": r"\Delta",
         "Λ": r"\Lambda", "Ω": r"\Omega", "Σ": r"\Sigma", "Φ": r"\Phi"}
REL = {"≈": r"\approx", "≥": r"\ge", "≤": r"\le", "±": r"\pm", "×": r"\times", "→": r"\to", "∝": r"\propto",
       "∞": r"\infty", "≪": r"\ll", "≫": r"\gg", "≳": r"\gtrsim", "≲": r"\lesssim", "∈": r"\in", "≡": r"\equiv", "∼": r"\sim", "√": r"\sqrt"}
SUP = {"⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
SUB = {"₀": "0", "₁": "1", "₂": "2", "₃": "3"}
VAR_BASE = "VGPkxyuRITEALNCbrgspnwqmd"
VAR_SUB = {"set", "reset", "crit", "max", "on", "off", "s", "l", "d", "p", "c", "c2", "P", "P0", "p0", "d0", "j",
           "t", "th", "in", "e", "mem", "relax", "decay", "unit", "f", "fo", "fc", "cw", "cf", "w", "j,t", "0", "1",
           "2", "b", "i", "k", "n", "a", "jn", "ref", "min", "y", "x", "z", "eff", "V", "sd", "ln", "lo", "hi",
           "probe", "p0,d0", "p,d", "ON", "OFF", "e", "ON,j"}


def sub_tex(s):
    s = s.strip("{}")
    return s if len(s) == 1 or s.isdigit() or "," in s else r"\mathrm{" + s + "}"


def subscript(s):
    t = sub_tex(s)
    return "_" + t if len(t) == 1 else "_{" + t + "}"


def math_atoms(text):
    # Greek (+ optional _subscript), e.g. τ_s, Λ_crit, η_p
    def greek(m):
        g, sub = m.group(1), m.group(2)
        out = GREEK[g] + (subscript(sub) if sub else "")
        return "\x00" + out + "\x01"
    text = re.sub(r"([" + "".join(GREEK) + r"])(?:_(\{[^}]+\}|[A-Za-z0-9]+(?:,[A-Za-z0-9]+)?))?", greek, text)

    # variables with a whitelisted subscript, as whole words: V_set, G_on, k_p0, P_c
    def var(m):
        base, sub = m.group(1), m.group(2)
        if sub.strip("{}") not in VAR_SUB:
            return m.group(0)
        return "\x00" + base + subscript(sub) + "\x01"
    text = re.sub(r"(?<![\w.\-])([" + VAR_BASE + r"])_(\{[^}]+\}|[A-Za-z0-9]+(?:,[A-Za-z0-9]+)?)(?![\w\-(]|\.\w|/\w+\.)",
                  var, text)
    # powers: 10⁻³, x², 10^-3 (keep units like mm² -> mm$^2$)
    def sup(m):
        return "\x00^{" + "".join(SUP[c] for c in m.group(1)) + "}\x01"
    text = re.sub(r"([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)", sup, text)
    text = re.sub(r"([₀₁₂₃]+)", lambda m: "\x00_{" + "".join(SUB[c] for c in m.group(1)) + "}\x01", text)
    for k, v in REL.items():
        text = text.replace(k, "\x00" + v + " \x01" if k == "√" else "\x00" + v + "\x01")
    # scientific notation 2.56e-6 -> 2.56 \times 10^{-6}, 1e-3 -> 10^{-3}
    def sci(m):
        mant, exp = m.group(1), int(m.group(2))
        return "\x00" + ("" if mant == "1" else mant + " \\times ") + "10^{" + str(exp) + "}\x01"
    text = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)e([+-]?\d+)(?![\w.])", sci, text)
    # ASCII multiplication between numbers: "3 x 2", "4x4" -> 3 \times 2
    text = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?) ?x ?(\d+(?:\.\d+)?)(?![\w.])",
                  lambda m: "\x00" + m.group(1) + " \\times " + m.group(2) + "\x01", text)
    text = re.sub(r"(?<![\w.])(\d+) x (?=[a-z])", lambda m: "\x00" + m.group(1) + " \\times\x01 ", text)
    text = re.sub(r"(?<=[a-z]) x (?=\d)", " \x00\\\\times\x01 ", text)          # "3 networks x 2 replicates"
    # ASCII powers 10^4, 10^-3
    text = re.sub(r"(?<![\w.])(\d+)\^(-?\d+)(?![\w.])", lambda m: "\x00" + m.group(1) + "^{" + m.group(2) + "}\x01", text)
    # abs(V) and escaped pipes \|V\|
    text = re.sub(r"\babs\(([A-Za-z_]+)\)", lambda m: "\x00\\lvert " + m.group(1) + " \\rvert\x01", text)
    text = re.sub(r"\\\|([A-Za-z_]+)\\\|", lambda m: "\x00\\lvert " + m.group(1) + " \\rvert\x01", text)
    return text


def finish(text):
    # glue: number or base right before a power (10^{-3}); merge adjacent atoms joined by spaces/operators/numbers
    text = re.sub(r"(\d+(?:\.\d+)?)\x00(\^\{[^\x01]*\})\x01", "\x00\\1\\2\x01", text)
    text = re.sub(r"([A-Za-z])\x00(\^\{[^\x01]*\})\x01", lambda m: "\x00" + m.group(1) + m.group(2) + "\x01"
                  if False else m.group(1) + "\x00" + m.group(2) + "\x01", text)
    # merge: atom [space/op/number...] atom -> one math span
    RELS = ("\\approx", "\\ge", "\\le", "\\pm", "\\times", "\\ll", "\\gg", "\\gtrsim", "\\lesssim",
            "\\propto", "\\to", "\\infty")

    def join(m):
        a, gap, b = m.group(1), m.group(2), m.group(3)
        if gap.strip() or a.rstrip().endswith(RELS) or b.lstrip().startswith(RELS) or not gap:
            return "\x00" + a + gap + b + "\x01"
        return m.group(0)
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\x00([^\x00\x01]*)\x01([ \t]*(?:[=+\-/*<>][ \t]*)?)\x00([^\x00\x01]*)\x01", join, text)
    # atom followed by a number: "\ge 0.8" -> keep number inside
    NUM = r"-?\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?"
    END = r"(?=[\s,;)|*]|\.(?!\d)|$)"
    REL_END = ("\\approx", "\\ge", "\\le", "\\pm", "\\times", "\\ll", "\\gg", "\\gtrsim", "\\lesssim", "\\propto")

    def after(m):
        atom, gap, op, num = m.group(1), m.group(2), m.group(3), m.group(4)
        if op or atom.rstrip().endswith(REL_END):
            return "\x00" + atom + gap + op + num + "\x01"
        return m.group(0)
    text = re.sub(r"\x00([^\x01]*)\x01([ \t]*)((?:[=<>][ \t]*)?)(" + NUM + r"(?:[ \t]?%)?)" + END, after, text)
    # atom preceded by a free-standing number: "0.8 \pm 0.1" (not "NARMA-10")
    text = re.sub(r"(?:(?<=^)|(?<=[\s(|*]))(\d+(?:\.\d+)?)[ \t]*([/*]?)[ \t]*\x00",
                  lambda m: "\x00" + m.group(1) + (m.group(2) if m.group(2) else " "), text)
    # number ranges inside math: en dash, not minus
    text = re.sub(r"\x00([^\x01]*)\x01", lambda m: "\x00" + re.sub(r"(\d)-(\d)", r"\1\\text{–}\2", m.group(1)) + "\x01", text)
    # escape % inside math
    text = re.sub(r"\x00([^\x01]*)\x01", lambda m: "\x00" + re.sub(r"(?<!\\)%", r"\\%", m.group(1)) + "\x01", text)
    return text.replace("\x00", "$").replace("\x01", "$")


def convert(src):
    out, in_code = [], False
    for line in src.split("\n"):
        if line.lstrip().startswith("```") or line.strip() == "$$":
            in_code = not in_code
            out.append(line)
            continue
        if in_code or line.lstrip().startswith("$$"):
            out.append(line)
            continue
        parts = re.split(r"(`[^`]*`|\]\([^)]*\)|<[^>]+>|https?://\S+|\$[^$]+\$)", line)
        new = "".join(p if i % 2 else finish(math_atoms(p)) for i, p in enumerate(parts))
        out.append(new)
    return "\n".join(out)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        src = open(path, encoding="utf8").read()
        dst = convert(src)
        if "--write" in sys.argv[:1] or os.environ.get("WRITE"):
            open(path, "w", encoding="utf8").write(dst)
        else:
            import difflib
            for l in difflib.unified_diff(src.split("\n"), dst.split("\n"), lineterm="", n=0):
                print(l)
