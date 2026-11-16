"""EBM-only inference adapter for the paper's final 60-feature model."""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

from config import FEATURE_NAMES

logger = logging.getLogger(__name__)

MODEL_NAME = "EBM"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "ebm_revision_artifact.joblib"
SYNDROME_LABELS = [
    "气阴两虚",
    "痰热互结",
    "肝肾阴虚",
    "热盛伤津",
    "肝胃郁热",
    "阴阳两虚",
]


def _validate_artifact(artifact, artifact_path):
    if not isinstance(artifact, dict):
        raise ValueError(f"EBM 工件格式无效：{artifact_path}")
    required = {
        "primary_model",
        "model_name",
        "feature_names",
        "class_order",
        "metadata",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(f"EBM 工件缺少字段：{missing}")
    if artifact["model_name"] != MODEL_NAME:
        raise ValueError("部署工件必须为论文最终 EBM 主模型")

    model_features = [str(name) for name in artifact["feature_names"]]
    if len(model_features) != 60 or len(set(model_features)) != 60:
        raise ValueError("EBM 工件必须包含 60 个唯一特征")
    if set(model_features) != set(FEATURE_NAMES):
        missing_in_model = sorted(set(FEATURE_NAMES) - set(model_features))
        extra_in_model = sorted(set(model_features) - set(FEATURE_NAMES))
        raise ValueError(
            "EBM 工件与部署特征目录不一致："
            f"缺少 {missing_in_model}，多出 {extra_in_model}"
        )
    if list(artifact["class_order"]) != SYNDROME_LABELS:
        raise ValueError(
            f"EBM 证型顺序与论文口径不一致：{list(artifact['class_order'])}"
        )
    return artifact


def load_model_artifact(model_path=None):
    """加载论文 EBM 工件；缺失或损坏时直接失败，不做模拟模型兜底。"""
    artifact_path = Path(model_path) if model_path else MODEL_PATH
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"未找到论文 EBM 工件：{artifact_path}。"
            "请部署 models/ebm_revision_artifact.joblib 后重试。"
        )
    try:
        artifact = joblib.load(artifact_path)
    except Exception as exc:
        logger.exception("EBM 工件加载失败：%s", artifact_path)
        raise RuntimeError(f"EBM 工件加载失败：{artifact_path}") from exc
    return _validate_artifact(artifact, artifact_path)


def _normalize_feature_levels(feature_levels):
    if not isinstance(feature_levels, dict):
        raise ValueError("EBM 输入必须是特征名到 1-5 级程度的映射")
    missing = [name for name in FEATURE_NAMES if name not in feature_levels]
    if missing:
        raise ValueError(f"缺少特征等级：{missing}")

    levels = []
    for name in FEATURE_NAMES:
        value = feature_levels[name]
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"特征“{name}”的等级必须是 1-5 的整数") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"特征“{name}”的等级必须是 1-5 的整数")
        level = int(numeric)
        if level < 1 or level > 5:
            raise ValueError(f"特征“{name}”的等级必须在 1-5 之间")
        levels.append(level)
    return np.asarray([levels], dtype=np.int8)


def levels_to_binary(feature_levels):
    """
    将界面五级程度转换为论文 EBM 的入模二值。

    1 级“无”映射为 0；2-5 级映射为 1。模型入模不使用中间连续值。
    """
    levels = _normalize_feature_levels(feature_levels)
    return (levels >= 2).astype(np.int8)


def _artifact_ordered_matrix(artifact, binary_display):
    artifact_index = {
        str(name): idx for idx, name in enumerate(artifact["feature_names"])
    }
    display_index = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
    artifact_rows_for_display = np.asarray(
        [artifact_index[name] for name in FEATURE_NAMES], dtype=int
    )
    display_rows_for_artifact = np.asarray(
        [display_index[str(name)] for name in artifact["feature_names"]], dtype=int
    )
    model_matrix = binary_display[:, display_rows_for_artifact]
    return model_matrix, artifact_rows_for_display


def _predict_probabilities(model, model_matrix):
    raw = np.asarray(model.predict_proba(model_matrix), dtype=float)
    ordered = np.zeros((1, len(SYNDROME_LABELS)), dtype=float)
    for source_idx, class_idx in enumerate(np.asarray(model.classes_, dtype=int)):
        ordered[:, int(class_idx)] = raw[:, source_idx]
    row_sum = ordered.sum(axis=1)
    if np.any(row_sum <= 0):
        raise RuntimeError("EBM 返回了无效的证型概率")
    return ordered / row_sum[:, None]


def _native_local_scores(model, model_matrix):
    explanation = model.explain_local(model_matrix, name="prediction")
    specific = explanation._internal_obj["specific"][0]
    scores = np.asarray(specific["scores"], dtype=float)
    intercepts = np.asarray(specific["extra"]["scores"][0], dtype=float)
    expected_shape = (len(FEATURE_NAMES), len(SYNDROME_LABELS))
    if scores.shape != expected_shape:
        raise RuntimeError(
            f"EBM 局部贡献维度异常：{scores.shape}，应为 {expected_shape}"
        )
    if intercepts.shape != (len(SYNDROME_LABELS),):
        raise RuntimeError(f"EBM 截距维度异常：{intercepts.shape}")
    return scores, intercepts


def _softmax(logits):
    shifted = np.asarray(logits, dtype=float) - np.max(logits)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def predict_syndrome(model, feature_levels):
    """执行论文 EBM 推理、概率排序与原生局部贡献计算。"""
    binary_display = levels_to_binary(feature_levels)
    artifact = model if isinstance(model, dict) else None
    primary_model = artifact["primary_model"] if artifact else model
    if artifact:
        model_matrix, artifact_rows_for_display = _artifact_ordered_matrix(
            artifact, binary_display
        )
    else:
        model_matrix = binary_display
        artifact_rows_for_display = np.arange(len(FEATURE_NAMES))
    display_levels = np.asarray(
        [[int(feature_levels[name]) for name in FEATURE_NAMES]], dtype=np.int8
    )

    probabilities = _predict_probabilities(primary_model, model_matrix)[0]
    predicted_idx = int(np.argmax(probabilities))
    scores, intercepts = _native_local_scores(primary_model, model_matrix)
    reconstructed = _softmax(intercepts + scores.sum(axis=0))
    reconstruction_error = float(np.max(np.abs(reconstructed - probabilities)))
    if reconstruction_error > 1e-6:
        raise RuntimeError(
            "EBM 原生局部贡献无法重建概率，"
            f"最大绝对误差为 {reconstruction_error:.3g}"
        )

    display_scores = scores[artifact_rows_for_display, :]
    contributions = {
        syndrome: display_scores[:, class_idx].copy()
        for class_idx, syndrome in enumerate(SYNDROME_LABELS)
    }
    return {
        "model": MODEL_NAME,
        "predicted_syndrome": SYNDROME_LABELS[predicted_idx],
        "probabilities": {
            syndrome: float(probabilities[class_idx])
            for class_idx, syndrome in enumerate(SYNDROME_LABELS)
        },
        "confidence": float(probabilities[predicted_idx]),
        "contributions": contributions,
        "feature_levels": display_levels,
        "binary_features": binary_display,
        "intercepts": {
            syndrome: float(intercepts[class_idx])
            for class_idx, syndrome in enumerate(SYNDROME_LABELS)
        },
        "reconstruction_error": reconstruction_error,
    }


if __name__ == "__main__":
    loaded_artifact = load_model_artifact()
    default_levels = {name: 1 for name in FEATURE_NAMES}
    check = predict_syndrome(loaded_artifact, default_levels)
    print("EBM artifact check success")
    print("features:", len(FEATURE_NAMES))
    print("predicted syndrome:", check["predicted_syndrome"])
    print("reconstruction error:", check["reconstruction_error"])
