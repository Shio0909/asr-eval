"""汇总多域三臂 VAD 长音频结果，并给出带硬门槛的综合选择。

输入文件名约定为 ``<arm>__<dataset>.json``，内容是 ``score.py`` 的结果。
程序不会重新调用模型；可在 XCOMET 后补完成后重复生成报告。
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


EXPECTED_ARMS = {
    "online_old",
    "local_impl_old_params",
    "local_impl_new_params",
}

ARM_ALIASES = {
    "A_online_old": "online_old",
    "B_local_old": "local_impl_old_params",
    "C_local_new": "local_impl_new_params",
}

COMPONENTS = {
    "quality": (
        ("xcomet", "higher"),
        ("chrF", "higher"),
        ("source_asr_error_rate", "lower"),
        ("term_recall", "higher"),
    ),
    "latency": (
        ("stable_ttfb_p50_s", "lower"),
        ("e2e_p95_ms", "lower"),
        ("finish_tail_p95_s", "lower"),
        ("long_dal_ca_ms", "lower"),
    ),
    "stability": (
        ("fail_rate", "lower"),
        ("retry_sample_rate", "lower"),
        ("segment_error_rate", "lower"),
        ("short_segment_rate", "lower"),
        ("anomaly_rate", "lower"),
    ),
    "cost": (
        ("requests_per_minute", "lower"),
        ("inference_p95_ms", "lower"),
    ),
}

COMPONENT_WEIGHTS = {
    "quality": 0.50,
    "latency": 0.25,
    "stability": 0.15,
    "cost": 0.10,
}


def _mean(values):
    values = [float(value) for value in values if isinstance(value, (int, float))]
    return round(statistics.mean(values), 4) if values else None


def _flatten(path: Path) -> tuple[dict, list[dict]]:
    first, separator, second = path.stem.partition("__")
    if not separator:
        raise ValueError(f"结果文件名缺少 '<arm>__<dataset>'：{path.name}")
    if first in EXPECTED_ARMS:
        arm, dataset = first, second
    elif second in ARM_ALIASES:
        arm, dataset = ARM_ALIASES[second], first
    else:
        arm, dataset = first, second
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary") or {}
    vad = summary.get("simult_vad") or {}
    n_requests = vad.get("n_requests") or 0
    anomaly_count = sum(
        int(value) for value in (vad.get("anomaly_counts") or {}).values()
        if isinstance(value, (int, float))
    )
    row = {
        "arm": arm,
        "dataset": dataset,
        "file": str(path),
        "n_total": summary.get("n_total"),
        "n_ok": summary.get("n_ok"),
        "low_coverage": bool(summary.get("low_coverage")),
        "fail_rate": summary.get("fail_rate"),
        "retry_sample_rate": summary.get("retry_sample_rate"),
        "xcomet": summary.get("xcomet"),
        "chrF": summary.get("chrF"),
        "BLEU": summary.get("BLEU"),
        "source_asr_error_rate": summary.get("source_asr_error_rate"),
        "term_recall": summary.get("term_recall"),
        "numeric_literal_recall": summary.get("numeric_literal_recall"),
        "date_literal_recall": summary.get("date_literal_recall"),
        "literal_omission_proxy_rate": summary.get("literal_omission_proxy_rate"),
        "literal_addition_proxy_rate": summary.get("literal_addition_proxy_rate"),
        "length_ratio": summary.get("translation_length_ratio_mean"),
        "stable_ttfb_p50_s": summary.get("stable_ttfb_p50_s"),
        "finish_tail_p95_s": summary.get("finish_tail_p95_s"),
        "long_dal_ca_ms": summary.get("long_dal_ca_ms"),
        "audio_minutes": vad.get("audio_minutes"),
        "segments_per_minute": vad.get("segments_per_minute"),
        "requests_per_minute": vad.get("requests_per_minute"),
        "short_segment_rate": vad.get("segment_lt_2s_rate"),
        "over_hard_max_rate": vad.get("segment_gt_hard_max_rate"),
        "reason_coverage": vad.get("segment_reason_coverage"),
        "e2e_p95_ms": vad.get("e2e_p95_ms"),
        "inference_p95_ms": vad.get("inference_p95_ms"),
        "segment_error_rate": (
            vad.get("segment_error_count", 0) / n_requests if n_requests else None
        ),
        "anomaly_rate": anomaly_count / n_requests if n_requests else None,
        "anomaly_count": anomaly_count,
    }
    anomalies = []
    for sample in payload.get("samples") or []:
        for anomaly in ((sample.get("simult") or {}).get("anomalies") or []):
            anomalies.append({
                "arm": arm,
                "dataset": dataset,
                "sample_id": sample.get("id"),
                **anomaly,
            })
    return row, anomalies


def _rank_component(rows: list[dict], metrics) -> dict[str, float | None]:
    earned = defaultdict(list)
    datasets = sorted({row["dataset"] for row in rows})
    for dataset in datasets:
        group = [row for row in rows if row["dataset"] == dataset]
        for key, direction in metrics:
            observed = [(row["arm"], row.get(key)) for row in group
                        if isinstance(row.get(key), (int, float))]
            if len(observed) < 2:
                continue
            ordered = sorted(observed, key=lambda item: item[1], reverse=direction == "higher")
            denominator = max(len(ordered) - 1, 1)
            for arm, value in observed:
                positions = [index for index, (_, candidate) in enumerate(ordered)
                             if candidate == value]
                rank = statistics.mean(positions)
                earned[arm].append(1 - rank / denominator)
    arms = {row["arm"] for row in rows}
    return {arm: _mean(earned.get(arm, [])) for arm in arms}


def _guardrails(rows: list[dict]) -> dict[str, list[str]]:
    reasons = defaultdict(list)
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
        if row["low_coverage"] or (row.get("fail_rate") or 0) > 0.05:
            reasons[row["arm"]].append(f"{row['dataset']}: 成功覆盖率低于 95%")
        if (row.get("retry_sample_rate") or 0) > 0.05:
            reasons[row["arm"]].append(f"{row['dataset']}: 超过 5% 样本发生重试")
        if (row.get("over_hard_max_rate") or 0) > 0.01:
            reasons[row["arm"]].append(f"{row['dataset']}: 超过 1% 分段越过 hard max")
    for dataset, group in by_dataset.items():
        chrfs = [row["chrF"] for row in group if isinstance(row.get("chrF"), (int, float))]
        asr_errors = [row["source_asr_error_rate"] for row in group
                      if isinstance(row.get("source_asr_error_rate"), (int, float))]
        best_chrf = max(chrfs) if chrfs else None
        best_asr = min(asr_errors) if asr_errors else None
        for row in group:
            if best_chrf is not None and isinstance(row.get("chrF"), (int, float)):
                if best_chrf - row["chrF"] > 1.5:
                    reasons[row["arm"]].append(
                        f"{dataset}: chrF 比该域最佳低 {best_chrf - row['chrF']:.2f}"
                    )
            if best_asr is not None and isinstance(row.get("source_asr_error_rate"), (int, float)):
                if row["source_asr_error_rate"] - best_asr > 0.015:
                    reasons[row["arm"]].append(
                        f"{dataset}: 源文错误率比该域最佳高 "
                        f"{row['source_asr_error_rate'] - best_asr:.2%}"
                    )
    return {arm: values for arm, values in reasons.items()}


def build_report(result_dir: str | Path) -> dict:
    result_dir = Path(result_dir)
    rows, anomalies = [], []
    for path in sorted(result_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("summary"), dict)
            or "infer_file" not in payload["summary"]
        ):
            continue
        row, found = _flatten(path)
        rows.append(row)
        anomalies.extend(found)
    arms = sorted({row["arm"] for row in rows})
    datasets_by_arm = {
        arm: sorted(row["dataset"] for row in rows if row["arm"] == arm)
        for arm in arms
    }
    complete = set(arms) == EXPECTED_ARMS and len({tuple(v) for v in datasets_by_arm.values()}) == 1
    component_scores = {
        component: _rank_component(rows, metrics)
        for component, metrics in COMPONENTS.items()
    }
    guardrail_failures = _guardrails(rows)
    arm_summaries = {}
    for arm in arms:
        component = {name: values.get(arm) for name, values in component_scores.items()}
        available = {name: value for name, value in component.items() if value is not None}
        weight_total = sum(COMPONENT_WEIGHTS[name] for name in available)
        composite = (
            sum(available[name] * COMPONENT_WEIGHTS[name] for name in available) / weight_total
            if weight_total else None
        )
        arm_rows = [row for row in rows if row["arm"] == arm]
        arm_summaries[arm] = {
            "eligible": arm not in guardrail_failures,
            "guardrail_failures": guardrail_failures.get(arm, []),
            "component_scores": component,
            "composite_score": round(composite, 4) if composite is not None else None,
            "datasets": len(arm_rows),
            "audio_minutes": _mean(row.get("audio_minutes") for row in arm_rows),
            "mean_chrF": _mean(row.get("chrF") for row in arm_rows),
            "mean_source_asr_error_rate": _mean(
                row.get("source_asr_error_rate") for row in arm_rows
            ),
            "mean_requests_per_minute": _mean(
                row.get("requests_per_minute") for row in arm_rows
            ),
            "anomaly_count": sum(row["anomaly_count"] for row in arm_rows),
        }
    eligible = [
        (arm, data["composite_score"])
        for arm, data in arm_summaries.items()
        if data["eligible"] and data["composite_score"] is not None
    ]
    winner = max(eligible, key=lambda item: item[1])[0] if complete and eligible else None
    return {
        "complete": complete,
        "winner": winner,
        "selection_rule": {
            "component_weights": COMPONENT_WEIGHTS,
            "quality_guardrails": {
                "coverage_min": 0.95,
                "retry_sample_rate_max": 0.05,
                "hard_max_exceed_rate_max": 0.01,
                "per_dataset_chrf_gap_max": 1.5,
                "per_dataset_source_error_gap_max": 0.015,
            },
        },
        "datasets_by_arm": datasets_by_arm,
        "arms": arm_summaries,
        "results": rows,
        "anomalies": anomalies,
        "observability": {
            "segment_reason_coverage": _mean(row.get("reason_coverage") for row in rows),
            "xcomet_result_coverage": round(
                sum(isinstance(row.get("xcomet"), (int, float)) for row in rows) / len(rows), 4
            ) if rows else 0,
            "token_usage": "unavailable unless the service adds it to si_segment_done.timing",
        },
    }


def write_markdown(report: dict, path: str | Path) -> None:
    lines = ["# VAD 长音频三臂全量对比", ""]
    if report["complete"]:
        lines.append(f"综合最优：`{report['winner'] or '无方案通过硬门槛'}`")
    else:
        lines.append("状态：结果尚未齐全，暂不选优。")
    lines.extend(["", "| 方案 | 硬门槛 | 综合分 | 质量 | 延迟 | 稳定性 | 成本 | 异常 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for arm, data in sorted(report["arms"].items()):
        scores = data["component_scores"]
        cell = lambda name: "—" if scores.get(name) is None else f"{scores[name]:.3f}"
        composite_cell = (
            "—" if data["composite_score"] is None else f"{data['composite_score']:.3f}"
        )
        lines.append(
            f"| {arm} | {'通过' if data['eligible'] else '不通过'} | "
            f"{composite_cell} | "
            f"{cell('quality')} | {cell('latency')} | {cell('stability')} | "
            f"{cell('cost')} | {data['anomaly_count']} |"
        )
    lines.extend(["", "## 硬门槛失败原因", ""])
    any_failure = False
    for arm, data in sorted(report["arms"].items()):
        for reason in data["guardrail_failures"]:
            any_failure = True
            lines.append(f"- `{arm}`：{reason}")
    if not any_failure:
        lines.append("- 无")
    lines.extend(["", "## 单独记录的异常", ""])
    if report["anomalies"]:
        for anomaly in report["anomalies"]:
            detail = json.dumps(anomaly, ensure_ascii=False, sort_keys=True)
            lines.append(f"- `{anomaly.get('arm')}/{anomaly.get('dataset')}/{anomaly.get('sample_id')}`：{detail}")
    else:
        lines.append("- 无")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()
    report = build_report(args.results_dir)
    Path(args.out_json).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(report, args.out_md)
    print(json.dumps({"complete": report["complete"], "winner": report["winner"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
