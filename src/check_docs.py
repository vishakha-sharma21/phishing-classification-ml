"""Check that key numbers quoted in README.md, reports/report.md and reports/viva_qa.md
match reports/results.json and reports/eda_results.json.

Run with ``python -m src.check_docs``; exits with status 1 on any mismatch.
"""

from __future__ import annotations

import json
import sys

from src.config import PROJECT_ROOT, REPORTS_DIR


def f3(x: float) -> str:
    return f"{x:.3f}"


def expected_strings() -> dict[str, list[str]]:
    r = json.loads((REPORTS_DIR / "results.json").read_text(encoding="utf-8"))
    e = json.loads((REPORTS_DIR / "eda_results.json").read_text(encoding="utf-8"))
    ev = r["evaluation"]
    t, ci, cmp_ = ev["test"], ev["test_bootstrap_ci"], ev["paired_comparisons_test"]
    lr = t["Custom LR"]
    d = cmp_["Custom LR vs Best single feature"]["f1"]
    sel_lr = r["logistic_regression"]["selected"]
    dup = ev["duplicate_diagnostic"]
    common = [
        f"{f3(lr['f1'])} [{f3(ci['Custom LR']['f1']['low'])}, {f3(ci['Custom LR']['f1']['high'])}]",
        f"{f3(lr['roc_auc'])} [{f3(ci['Custom LR']['roc_auc']['low'])}, {f3(ci['Custom LR']['roc_auc']['high'])}]",
        f"+{f3(d['mean'])} [+{f3(d['low'])}, +{f3(d['high'])}]",
    ]
    report = common + [
        f"| {lr['tp']} | {lr['fp']} | {lr['fn']} | {lr['tn']} |",
        f"threshold {sel_lr['threshold']:.2f}",
        f"{sel_lr['iterations']:,} iterations",
        f"{r['preprocessing']['partition_sizes']['train']:,} / {r['preprocessing']['partition_sizes']['val']} / "
        f"{r['preprocessing']['partition_sizes']['test']}",
        f"{e['duplicate_leakage_diagnostic']['raw_val_overlap_percent']:.1f}%",
        f"{f3(dup['Custom LR raw-split val']['accuracy'])} vs {f3(dup['Custom LR clean val']['accuracy'])}",
    ]
    viva = [f"+{f3(d['mean'])} [+{f3(d['low'])}, +{f3(d['high'])}]", f"LR {sel_lr['threshold']:.2f}"]
    q = r["quick_models"]
    qci, qz = q["test_bootstrap_ci"], q["zones"]["thresholds"]["lr"]
    qd = q["paired_comparisons_test"]["Quick LR (8) vs Full LR (30)"]["f1"]
    quick = [f"{f3(q['test']['Quick LR (8)']['f1'])} [{f3(qci['Quick LR (8)']['f1']['low'])}, "
             f"{f3(qci['Quick LR (8)']['f1']['high'])}]"]
    report += quick + [f"low {qz['low']:.2f}, high {qz['high']:.2f}", f"ΔF1 {qd['mean']:+.3f}".replace("-", "−")]
    viva += [f"{qz['low']:.2f} and {qz['high']:.2f}"]
    common += quick
    return {"README.md": common, "reports/report.md": report, "reports/viva_qa.md": viva}


def main() -> int:
    bad = 0
    for doc, strings in expected_strings().items():
        text = (PROJECT_ROOT / doc).read_text(encoding="utf-8")
        for s in strings:
            ok = s in text
            bad += not ok
            print(f"{'OK  ' if ok else 'MISS'} {doc}: {s}")
    print(f"\n{bad} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
