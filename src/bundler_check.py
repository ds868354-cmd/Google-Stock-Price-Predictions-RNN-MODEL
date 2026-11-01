"""Standalone no-pickle/joblib guardrail -- self-contained copy of
verify_no_pickle_joblib() from the training pipeline, so CI doesn't need that module installed.
Usage: python bundler_check.py <directory>
"""
import os, re, sys

_BANNED_EXTENSIONS = {".pkl", ".pickle", ".joblib"}
_PICKLE_IMPORT_PATTERN = re.compile(
    r"^\s*(import\s+(pickle|joblib)\b|from\s+(pickle|joblib)\s+import)", re.MULTILINE)


def verify_no_pickle_joblib(directory):
    violations = []
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            fpath = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext in _BANNED_EXTENSIONS:
                violations.append("Banned artifact file: " + fpath)
                continue
            if ext == ".py":
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except OSError:
                    continue
                if _PICKLE_IMPORT_PATTERN.search(content):
                    violations.append("Banned import (pickle/joblib) in: " + fpath)
    if violations:
        raise SystemExit("Pickle/joblib guardrail failed:\n  " + "\n  ".join(violations))
    print("'" + directory + "' is clean (no pickle/joblib found).")


if __name__ == "__main__":
    verify_no_pickle_joblib(sys.argv[1] if len(sys.argv) > 1 else ".")
