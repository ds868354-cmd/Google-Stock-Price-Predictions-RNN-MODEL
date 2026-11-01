"""Model quality gate, run by ci.yml (job `test`) BEFORE any image is built. Standard library only.

Reads model_metrics.json (written by the training pipeline: metric numbers only, never data rows) and checks it
  1. against sane FLOORS for the model's task, then
  2. against your own targets in ci_thresholds.json.
Exit code 0 = pass, 1 = a gate failed, so a degenerate or regressed model never reaches the registry.

  classification : roc_auc >= 0.5 (better than a coin flip); f1 / precision / recall must be > 0 when reported
  regression     : r2 > 0 (better than always predicting the mean); rmse / mae / mse must be finite and >= 0
  other types    : (clustering, anomaly detection, RL, LLM ...) every reported metric must be a finite number

ci_thresholds.json (optional, edit it in git like any other config):
  {"enforce": true, "min": {"roc_auc": 0.95, "recall_positive": 0.8}, "max": {"rmse": 12.5}}
enforce=false turns every failure into a warning annotation (useful while a model is still being tuned).
"""
import json
import math
import os
import sys

METRICS_FILE = os.environ.get("MODEL_METRICS_FILE", "model_metrics.json")
THRESHOLDS_FILE = os.environ.get("CI_THRESHOLDS_FILE", "ci_thresholds.json")

CLASSIFICATION_FLOORS = (("roc_auc", ">=", 0.5), ("f1_positive", ">", 0.0), ("f1_weighted", ">", 0.0), ("f1_macro", ">", 0.0),
                         ("recall_positive", ">", 0.0), ("precision_positive", ">", 0.0))
REGRESSION_FLOORS = (("r2", ">", 0.0), ("test_r2", ">", 0.0), ("rmse", ">=", 0.0), ("mae", ">=", 0.0), ("mse", ">=", 0.0))
NO_TASK_MODEL_TYPES = ("UNSUPERVISED_ML", "ANOMALY_DETECTION", "TABULAR_RL", "RAG_LLM", "LLM_FINETUNE")


def _load(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _task_of(payload, metrics):
    task = str(payload.get("task") or "").strip().lower()
    if payload.get("model_type") in NO_TASK_MODEL_TYPES:
        return "other"
    if task.startswith("class") or task in ("binary", "multiclass", "multilabel"):
        return "classification"
    if task.startswith("regress") or task in ("forecast", "forecasting", "time_series"):
        return "regression"
    if any(k in metrics for k in ("r2", "test_r2", "rmse", "mae")):
        return "regression"
    if any(k in metrics for k in ("accuracy", "roc_auc", "f1_weighted", "f1_macro", "f1_positive")):
        return "classification"
    return "other"


def _passes(value, op, bound):
    if op == ">":
        return value > bound
    if op == "<=":
        return value <= bound
    return value >= bound


def main():
    payload = _load(METRICS_FILE)
    if payload is None:
        print("::warning::%s not found -- model quality gate skipped (re-run the training pipeline to create it)" % METRICS_FILE)
        return 0
    metrics = payload.get("metrics") or {}
    cfg = _load(THRESHOLDS_FILE) or {}
    enforce = bool(cfg.get("enforce", True))
    task = _task_of(payload, metrics)
    rows, failures = [], []

    def check(name, op, bound, kind):
        value = metrics.get(name)
        if value is None:
            return
        try:
            bound = float(bound)
        except (TypeError, ValueError):
            failures.append("%s: the bound %r in %s is not a number" % (name, bound, THRESHOLDS_FILE))
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            rows.append((name, str(value), kind + " finite", "FAIL"))
            failures.append("%s is not a finite number (%r)" % (name, value))
            return
        value = float(value)
        passed = _passes(value, op, bound)
        rows.append((name, "%.6g" % value, "%s %s %g" % (kind, op, bound), "ok" if passed else "FAIL"))
        if not passed:
            failures.append("%s = %.6g, required %s %g (%s)" % (name, value, op, bound, kind))

    floors = CLASSIFICATION_FLOORS if task == "classification" else REGRESSION_FLOORS if task == "regression" else ()
    for name, op, bound in floors:
        check(name, op, bound, "floor")
    if task == "other":
        for name in sorted(metrics):
            check(name, ">=", float("-inf"), "finite")
    for section, op in (("min", ">="), ("max", "<=")):
        for name, bound in sorted((cfg.get(section) or {}).items()):
            if name not in metrics:
                failures.append("%s asks for %s %s but model_metrics.json has no such metric" % (THRESHOLDS_FILE, section, name))
            else:
                check(name, op, bound, "target")

    print("Model quality gate | model_type=%s task=%s best_model=%s n_test=%s enforce=%s"
          % (payload.get("model_type"), task, payload.get("best_model_name"), payload.get("n_test"), enforce))
    for name, value, rule, status in rows:
        print("  %-30s %-14s %-24s %s" % (name, value, rule, status))
    if not rows:
        print("  (no metric matched a gate for this model type -- nothing to check)")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write("### Model quality gate (%s)\n\n| metric | value | rule | result |\n|---|---|---|---|\n" % task)
                for name, value, rule, status in rows:
                    f.write("| %s | %s | %s | %s |\n" % (name, value, rule, status))
        except OSError:
            pass
    if failures:
        level = "error" if enforce else "warning"
        for message in failures:
            print("::%s::model quality gate: %s" % (level, message))
        return 1 if enforce else 0
    print("Model quality gate passed (%d check(s))." % len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
