"""Notebook-compatible preprocessing, causal prediction and alarm policy."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import numpy as np
import pandas as pd

CLASSES = ["No leak", "LOCA", "LOCAC", "SGATR", "SGBTR", "SLBIC", "FLB", "LLB"]
LEAK_CLASSES = CLASSES[1:]
WHERE = {"No leak": "No leak indicated", "LOCA": "Hot-leg pipe", "LOCAC": "Cold-leg pipe", "SGATR": "Steam generator A tubes", "SGBTR": "Steam generator B tubes", "SLBIC": "Steam line", "FLB": "Feedwater line", "LLB": "Letdown line"}
SIGNALS = {"P": ("Primary pressure", "bar"), "TAVG": ("Coolant temperature", "°C"), "WRCA": ("Loop A flow", "t/h"), "WRCB": ("Loop B flow", "t/h"), "RM1": ("Containment radiation monitor", "CPM"), "RM2": ("Steam-line radiation monitor", "CPM"), "PSGA": ("Steam generator A pressure", "bar"), "WFWA": ("Feedwater A flow", "t/h")}


def tensorflow():
    import tensorflow as tf
    return tf


def restricted_joblib(path: Path):
    """Read the notebook's uncompressed joblib exports without arbitrary globals."""
    from joblib.numpy_pickle import NumpyUnpickler
    allowed = {
        ("sklearn.preprocessing._data", "StandardScaler"),
        ("joblib.numpy_pickle", "NumpyArrayWrapper"),
        ("numpy", "ndarray"), ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
    }
    class Reader(NumpyUnpickler):
        def find_class(self, module, name):
            if (module, name) not in allowed:
                raise ValueError(f"Unsupported object in {path.name}: {module}.{name}. Export a standard notebook model package.")
            return super().find_class(module, name)
    with path.open("rb") as handle:
        return Reader(str(path), handle, ensure_native_byte_order=True, mmap_mode=None).load()


def load_bundle(folder: Path):
    import zipfile
    meta = json.loads((folder / "manifest.json").read_text())
    config = restricted_joblib(folder / "config.pkl")
    scaler = restricted_joblib(folder / "scaler.pkl")
    if not isinstance(config, dict):
        raise ValueError("config.pkl must contain the notebook configuration dictionary.")
    features = config.get("feature_cols")
    if not isinstance(features, list) or not features or not all(isinstance(x, str) for x in features) or len(set(features)) != len(features):
        raise ValueError("Configuration needs a unique, ordered feature_cols list.")
    for key in ("time_steps", "stride"):
        if not isinstance(config.get(key), int) or isinstance(config[key], bool) or not 1 <= config[key] <= 10000:
            raise ValueError(f"Invalid {key} in config.pkl.")
    classes = meta.get("classes")
    if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes) or len(set(classes)) != len(classes) or set(classes) not in (set(CLASSES), set(LEAK_CLASSES)):
        raise ValueError("Declare the locator's exact output order: all seven supported leak classes, optionally including No leak.")
    if not isinstance(meta.get("sample_interval"), (int, float)) or isinstance(meta["sample_interval"], bool) or not 0 < meta["sample_interval"] < 86400:
        raise ValueError("A valid sampling interval in seconds is required.")
    if not isinstance(meta.get("threshold"), (int, float)) or isinstance(meta["threshold"], bool) or not 0 < meta["threshold"] <= 1 or not isinstance(meta.get("persistence"), int) or isinstance(meta["persistence"], bool) or not 1 <= meta["persistence"] <= 100:
        raise ValueError("Invalid alarm threshold or persistence.")
    from sklearn.preprocessing import StandardScaler
    if not isinstance(scaler, StandardScaler) or getattr(scaler, "n_features_in_", None) != len(features):
        raise ValueError("The StandardScaler does not match the configured feature count.")
    if hasattr(scaler, "feature_names_in_") and list(scaler.feature_names_in_) != features:
        raise ValueError("Scaler feature order differs from config.pkl.")
    if not np.isfinite(scaler.transform(np.zeros((1, len(features))))).all():
        raise ValueError("Scaler contains invalid parameters.")
    allowed_layers = {"Sequential", "InputLayer", "LSTM", "Dropout", "Dense"}
    def check_layers(node):
        if isinstance(node, dict):
            if (node.get("module") or "").startswith("keras.layers") and node.get("class_name") not in allowed_layers:
                raise ValueError(f"Unsupported layer: {node.get('class_name')}. This app supports the notebook's Sequential LSTM models.")
            if node.get("class_name") == "__lambda__":
                raise ValueError("Lambda model objects are not supported.")
            for v in node.values(): check_layers(v)
        elif isinstance(node, list):
            for v in node: check_layers(v)
    tf = tensorflow()
    nets = []
    for filename, width in [("lstm_leak_classifier.keras", 1), ("lstm_leak_location.keras", len(classes))]:
        with zipfile.ZipFile(folder / filename) as archive:
            if sum(x.file_size for x in archive.infolist()) > 256 * 1024**2:
                raise ValueError("Model archive expands beyond the supported limit.")
            model_config = json.loads(archive.read("config.json"))
            if model_config.get("class_name") != "Sequential":
                raise ValueError("Import the Sequential models exported by the supplied notebook.")
            check_layers(model_config)
        net = tf.keras.models.load_model(folder / filename, compile=False, safe_mode=True)
        if tuple(net.input_shape[1:]) != (config["time_steps"], len(features)) or net.output_shape[-1] != width:
            raise ValueError(f"{filename} expects ({config['time_steps']}, {len(features)}) inputs and {width} outputs; received {net.input_shape[1:]} and {net.output_shape[-1]} outputs. Check the selected locator format and class order.")
        nets.append(net)
    return {"meta": meta, "config": config, "scaler": scaler, "detector": nets[0], "locator": nets[1]}


def read_csv(raw: bytes, config: dict, interval: float):
    if len(raw) > 25 * 1024**2:
        raise ValueError("CSV limit is 25 MB per recording.")
    try:
        text = raw.decode("utf-8-sig")
        headers = next(csv.reader(io.StringIO(text)))
        if len(headers) != len(set(headers)):
            raise ValueError("CSV headers must be unique.")
        df = pd.read_csv(io.StringIO(text))
    except (UnicodeError, pd.errors.ParserError, StopIteration) as exc:
        raise ValueError("Use a UTF-8, comma-separated sensor CSV with a header row.") from exc
    required = ["TIME", *config["feature_cols"]]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError("Missing required sensors: " + ", ".join(missing))
    if len(df) < config["time_steps"]:
        raise ValueError(f"Need at least {config['time_steps']} readings; received {len(df)}.")
    if len(df) > 50000: raise ValueError("Limit is 50,000 readings per recording.")
    try: df[required] = df[required].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc: raise ValueError("TIME and required sensors must contain only numeric values.") from exc
    if not np.isfinite(df[required].to_numpy(dtype=float)).all():
        raise ValueError("Required readings contain blank, NaN or infinite values.")
    steps = np.diff(df.TIME.to_numpy(dtype=float))
    if not np.all(steps > 0): raise ValueError("TIME must increase strictly without duplicates.")
    if not np.allclose(steps, interval, rtol=1e-4, atol=1e-5):
        raise ValueError(f"Sampling interval must be {interval:g} seconds to match the model. No automatic resampling is applied.")
    return df


def fingerprint(df, features):
    # Canonical numeric values catch renamed/reformatted copies, including time shifts.
    values = np.ascontiguousarray(df[features].to_numpy(dtype="<f8"))
    return hashlib.sha256(values.tobytes()).hexdigest()


def windows(df, bundle):
    cfg = bundle["config"]
    scaled = bundle["scaler"].transform(df[cfg["feature_cols"]].to_numpy(dtype=np.float32)).astype(np.float32)
    starts = np.arange(0, len(df) - cfg["time_steps"] + 1, cfg["stride"])
    result = np.stack([scaled[i:i + cfg["time_steps"]] for i in starts])
    endpoints = df.TIME.to_numpy()[starts + cfg["time_steps"] - 1].astype(float)
    return result, endpoints


def training_windows(df, bundle, row):
    """Detector sees all clips; V3 leak-only locator sees the first six per recording."""
    x, times = windows(df, bundle)
    labels = np.full(len(x), CLASSES.index(row["label"]), dtype=np.int32)
    if row.get("onset") is not None:
        labels[times < row["onset"]] = 0
    eligible = np.ones(len(x), dtype=bool)
    if "No leak" not in bundle["meta"]["classes"]:
        eligible = (np.arange(len(x)) < 6) & (labels > 0)
    return x, labels, eligible


def scores_for(net, x):
    # Direct calls avoid a background tf.data threadpool for each replay.
    out = np.concatenate([np.asarray(net(x[i:i+128], training=False)) for i in range(0, len(x), 128)])
    if not np.isfinite(out).all() or np.any(out < -1e-6) or np.any(out > 1 + 1e-6):
        raise ValueError("Model returned invalid scores; expected probabilities in [0, 1].")
    return out


def alarm_series(times, probabilities, locations, threshold, persistence, class_names=None):
    class_names = CLASSES if class_names is None else class_names
    conditional = "No leak" not in class_names
    predictions, events = [], []
    hits, alarm = 0, False
    for i, (time, score) in enumerate(zip(times, probabilities)):
        loc = None if locations is None else locations[i]
        hits = hits + 1 if float(score) >= threshold else 0
        location_component = class_names[int(np.argmax(loc))] if loc is not None else "No leak"
        component = "No leak" if conditional and float(score) < threshold else location_component
        if hits == 1 and not alarm:
            events.append({"time": float(time), "kind": "elevated", "message": "Score crossed the alarm threshold", "component": component})
        if hits >= persistence and not alarm:
            alarm = True
            events.append({"time": float(time), "kind": "alarm", "message": "Model alarm confirmed", "component": component})
        predictions.append({"time": float(time), "score": float(score), "hits": int(hits), "alarm": alarm, "component": component, "location_component": location_component,
            "locations": {} if loc is None else {c: float(v) for c, v in zip(class_names, loc)}, "disagreement": False if conditional else bool((float(score) >= threshold) != (component != "No leak"))})
    return predictions, events


def infer(df, bundle, name):
    x, endpoints = windows(df, bundle)
    scores = scores_for(bundle["detector"], x).reshape(-1)
    meta = bundle["meta"]
    conditional = "No leak" not in meta["classes"]
    localization = None
    if conditional:
        predictions, events = alarm_series(endpoints, scores, None, meta["threshold"], meta["persistence"], meta["classes"])
        alarm_event = next((e for e in events if e["kind"] == "alarm"), None)
        # Confirm detection first. Vote on the recording's first three causal clips,
        # not the three clips that trigger the alarm and not subsequent clips.
        if alarm_event is not None and len(x) >= 3:
            loc = scores_for(bundle["locator"], x[:3])
            if not np.allclose(loc.sum(axis=1), 1, atol=1e-3):
                raise ValueError("Location model output must be a softmax distribution.")
            vote_indices = np.argmax(loc, axis=1)
            counts = np.bincount(vote_indices, minlength=len(meta["classes"]))
            contenders = np.flatnonzero(counts == counts.max())
            means = loc.mean(axis=0)
            # V3 uses pandas Series.mode()[0]: tied string labels sort alphabetically.
            winner = min(contenders.tolist(), key=lambda i: meta["classes"][i])
            component = meta["classes"][winner]
            available_at = max(alarm_event["time"], float(endpoints[2]))
            localization = {"policy":"First three recording clips, after confirmed alarm", "clip_times":endpoints[:3].tolist(), "votes":[meta["classes"][int(v)] for v in vote_indices], "tied":len(contenders)>1, "tie_break":"Alphabetical class name (V3 pandas mode)", "available_at":available_at}
            for prediction in predictions:
                if prediction["time"] >= available_at:
                    prediction.update(component=component, location_component=component, locations={c:float(v) for c,v in zip(meta["classes"],means)})
            if available_at == alarm_event["time"]:
                alarm_event["component"] = component
            else:
                events.append({"time":available_at, "kind":"location", "message":"Three-clip location vote available", "component":component})
    else:
        loc = scores_for(bundle["locator"], x)
        if not np.allclose(loc.sum(axis=1), 1, atol=1e-3):
            raise ValueError("Location model output must be a softmax distribution.")
        predictions, events = alarm_series(endpoints, scores, loc, meta["threshold"], meta["persistence"], meta["classes"])
    signals = [c for c in SIGNALS if c in df and c in bundle["config"]["feature_cols"]]
    signals += [c for c in bundle["config"]["feature_cols"] if c not in signals]
    signals = signals[:12]
    # Units must come from the dataset/model package; the notebook does not declare them.
    units = meta.get("signal_units", {})
    sensors = [{"key": c, "label": SIGNALS.get(c, (c, "dataset units"))[0], "unit": units.get(c, "dataset units")} for c in signals]
    return {"name": name, "illustrative": False, "model_id": meta["id"], "model_name": meta["name"], "location_classes": meta["classes"], "location_conditional": conditional, "localization": localization, "threshold": meta["threshold"], "persistence": meta["persistence"], "sample_interval": meta["sample_interval"], "time_steps": bundle["config"]["time_steps"], "stride": bundle["config"]["stride"], "sensors": sensors, "readings": df[["TIME", *signals]].rename(columns={"TIME":"time"}).to_dict(orient="records"), "predictions": predictions, "events": events, "row_count": len(df), "start": float(df.TIME.iloc[0]), "end": float(df.TIME.iloc[-1])}


def evaluate_runs(bundle, runs):
    tp=fn=fp=tn=correct=total=0
    delays=[]
    conditional = "No leak" not in bundle["meta"]["classes"]
    for df, row in runs:
        result = infer(df, bundle, row["name"])
        alarms = [e for e in result["events"] if e["kind"] == "alarm"]
        onset = row.get("onset")
        is_leak = row["label"] != "No leak"
        early = bool(alarms and onset is not None and alarms[0]["time"] < onset)
        detected = bool(alarms) and not early
        if is_leak:
            tp += int(detected); fn += int(not detected)
            if detected and onset is not None: delays.append(alarms[0]["time"]-onset)
            fp += int(early)
        else:
            fp += int(bool(alarms)); tn += int(not alarms)
        if conditional:
            if is_leak and detected and result["localization"] is not None:
                correct += int(result["predictions"][-1]["location_component"] == row["label"]); total += 1
        else:
            for p in result["predictions"]:
                expected = row["label"] if onset is None or p["time"] >= onset else "No leak"
                correct += int(p["location_component"] == expected); total += 1
    return {"recordings": len(runs), "detected_leak_runs": tp, "missed_leak_runs": fn, "false_alarm_runs": fp, "quiet_nonleak_runs": tn, "location_correct": correct, "location_total": total, "location_windows_correct": None if conditional else correct, "location_windows_total": None if conditional else total, "location_accuracy": correct/total if total else None, "location_evaluation_scope": "Detected leak recordings, first-three-clip vote" if conditional else "All labeled windows", "median_delay_seconds": float(np.median(delays)) if delays else None, "delay_detected_runs": len(delays)}
