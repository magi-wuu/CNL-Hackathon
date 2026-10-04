"""Local API for immutable model versions, recorded replay and continued training."""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .pipeline import CLASSES, evaluate_runs, fingerprint, infer, load_bundle, read_csv, scores_for, tensorflow, training_windows, windows

ROOT = Path(os.environ.get("REACTOR_STORAGE_DIR", Path(__file__).resolve().parents[1]))
MODELS, DATA = ROOT / "models", ROOT / "data"
MODELS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
LOCK = threading.RLock()
CACHE = {}
REQUIRED = {"lstm_leak_classifier.keras", "lstm_leak_location.keras", "scaler.pkl", "config.pkl"}


def canonical_filename(filename):
    # Version suffixes describe model versions, not different input roles.
    match = re.fullmatch(r"(lstm_leak_(?:classifier|location))(?:_v[0-9]+)?\.keras", filename)
    return match.group(1) + ".keras" if match else filename


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def ident(value):
    try:
        if str(uuid.UUID(value)) != value: raise ValueError()
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid identifier.")
    return value


def registry():
    return read_json(MODELS / "index.json", {"active": None, "versions": []})


def get_bundle(model_id):
    model_id = ident(model_id)
    with LOCK:
        if model_id not in {v["id"] for v in registry()["versions"]}:
            raise HTTPException(404, "Model version not found.")
        if model_id not in CACHE:
            try: CACHE[model_id] = load_bundle(MODELS / model_id)
            except Exception as exc: raise HTTPException(422, f"Cannot load model package: {exc}") from exc
        return CACHE[model_id]


def add_version(meta, bundle=None):
    with LOCK:
        index = registry()
        index["versions"].append(meta)
        write_json(MODELS / "index.json", index)
        if bundle is not None: CACHE[meta["id"]] = bundle


app = FastAPI(title="Reactor Control Panel", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"], allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.exception_handler(ValueError)
async def bad_input(_, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/api/health")
def health():
    return {"status": "ready", "active_model": registry()["active"], "classes": CLASSES}


@app.get("/api/models")
def models():
    return registry()


@app.post("/api/models/import")
async def import_model(files: list[UploadFile] = File(...), name: str = Form(...), sample_interval: float = Form(...), threshold: float = Form(.7), persistence: int = Form(3), class_order_confirmed: bool = Form(False), location_classes: str = Form(json.dumps(CLASSES))):
    if not class_order_confirmed:
        raise ValueError("Confirm the exact location-label order against the model’s training labels before importing.")
    if not name.strip() or len(name) > 80: raise ValueError("Use a model name of 1–80 characters.")
    model_id = str(uuid.uuid4())
    folder = MODELS / model_id
    folder.mkdir()
    try:
        incoming = {}
        source_files = {}
        total = 0
        for file in files:
            raw = await file.read(256 * 1024**2 + 1)
            total += len(raw)
            if total > 256 * 1024**2: raise ValueError("Model package upload limit is 256 MB.")
            filename = Path(file.filename or "").name
            if filename.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    if len(archive.infolist()) > 30 or sum(i.file_size for i in archive.infolist()) > 256 * 1024**2:
                        raise ValueError("Model package expands beyond supported limits.")
                    for entry in archive.infolist():
                        if entry.is_dir(): continue
                        if entry.filename.startswith("/") or ".." in Path(entry.filename).parts:
                            raise ValueError("Invalid path in model archive.")
                        basename = Path(entry.filename).name
                        key = canonical_filename(basename)
                        if key not in REQUIRED | {"manifest.json"}: continue
                        if key in incoming: raise ValueError(f"Duplicate package file: {key}")
                        incoming[key] = archive.read(entry)
                        source_files[key] = basename
            else:
                key = canonical_filename(filename)
                if key not in REQUIRED: raise ValueError(f"Unsupported file {filename}. Use the detector/locator notebook exports (version suffixes such as _v2 are accepted), scaler.pkl and config.pkl, or a ZIP package.")
                if key in incoming: raise ValueError(f"Duplicate package role: {key}. Choose one detector and one locator.")
                incoming[key] = raw
                source_files[key] = filename
        missing = REQUIRED - incoming.keys()
        if missing: raise ValueError("Model package is incomplete. Missing: " + ", ".join(sorted(missing)))
        original = json.loads(incoming.get("manifest.json", b"{}"))
        # Re-import preserves the original sampling/alarm contract when a manifest exists.
        meta = {"id": model_id, "name": name.strip(), "created": now(), "kind": "imported", "classes": original["classes"] if "classes" in original else json.loads(location_classes), "source_files": source_files,
                "sample_interval": original.get("sample_interval", sample_interval), "threshold": original.get("threshold", threshold), "persistence": original.get("persistence", persistence)}
        units = original.get("signal_units", {})
        if not isinstance(units, dict) or not all(isinstance(k, str) and isinstance(v, str) and len(v) <= 40 for k, v in units.items()):
            raise ValueError("signal_units must map sensor names to short unit strings.")
        meta["signal_units"] = units
        for filename in REQUIRED: (folder / filename).write_bytes(incoming[filename])
        write_json(folder / "manifest.json", meta)
        bundle = load_bundle(folder)
        meta["localization_policy"] = "first_three_recording_clips_after_alarm" if "No leak" not in meta["classes"] else "per_window"
        meta["localization_tie_break"] = "alphabetical_class_name"
        meta["locator_training_clip_limit"] = 6 if "No leak" not in meta["classes"] else None
        meta.update({"features": bundle["config"]["feature_cols"], "time_steps": bundle["config"]["time_steps"], "stride": bundle["config"]["stride"]})
        if original.get("evaluation"): meta["evaluation"] = original["evaluation"]
        write_json(folder / "manifest.json", meta)
        bundle["meta"] = meta
        add_version(meta, bundle)
        return meta
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        if isinstance(exc, (ValueError, HTTPException)): raise
        raise ValueError(f"Import failed: {exc}. Use standard uncompressed notebook exports; custom layers and legacy H5 models are unsupported.") from exc


@app.post("/api/models/{model_id}/activate")
def activate(model_id: str):
    get_bundle(model_id)
    with LOCK:
        index = registry(); index["active"] = model_id
        write_json(MODELS / "index.json", index)
    return index


@app.get("/api/models/{model_id}/export")
def export(model_id: str):
    get_bundle(model_id)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(REQUIRED | {"manifest.json"}): archive.write(MODELS / model_id / name, name)
    return Response(stream.getvalue(), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="reactor-model-{model_id[:8]}.zip"'})


@app.get("/api/models/{model_id}/template")
def template(model_id: str):
    import csv
    cfg = get_bundle(model_id)["config"]
    out = io.StringIO(); csv.writer(out).writerow(["TIME", *cfg["feature_cols"]])
    return Response(out.getvalue(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="sensor-template.csv"'})


@app.post("/api/recordings")
async def recording(file: UploadFile = File(...), model_id: str = Form(...)):
    bundle = get_bundle(model_id)
    raw = await file.read(25 * 1024**2 + 1)
    df = read_csv(raw, bundle["config"], bundle["meta"]["sample_interval"])
    recording_id = str(uuid.uuid4())
    meta = {"id": recording_id, "name": Path(file.filename or "recording.csv").name[:150], "model_id": model_id,
            "rows": len(df), "start": float(df.TIME.iloc[0]), "end": float(df.TIME.iloc[-1]), "fingerprint": fingerprint(df, bundle["config"]["feature_cols"])}
    (DATA / f"{recording_id}.csv").write_bytes(raw)
    write_json(DATA / f"{recording_id}.json", meta)
    return meta


def read_recording(recording_id, bundle):
    recording_id = ident(recording_id)
    path = DATA / f"{recording_id}.csv"
    if not path.exists(): raise HTTPException(404, "Recording not found.")
    return read_csv(path.read_bytes(), bundle["config"], bundle["meta"]["sample_interval"])


@app.get("/api/recordings/{recording_id}/replay")
def replay(recording_id: str, model_id: str):
    bundle = get_bundle(model_id)
    df = read_recording(recording_id, bundle)
    meta = read_json(DATA / f"{recording_id}.json", {})
    return infer(df, bundle, meta.get("name", "Recording"))


class TrainingRow(BaseModel):
    id: str
    group: str = Field(min_length=1, max_length=100)
    split: str
    label: str
    onset: float | None = None
    origin: str = "new"


class TrainingRequest(BaseModel):
    model_id: str
    name: str = Field(min_length=1, max_length=80)
    rows: list[TrainingRow] = Field(min_length=3, max_length=40)
    epochs: int = Field(default=5, ge=1, le=20)
    learning_rate: float = Field(default=0.00001, ge=0.0000001, le=0.001)
    independent_holdout_confirmed: bool = False


def validate_training(request, bundle):
    if not request.independent_holdout_confirmed:
        raise ValueError("Confirm validation/test groups were not used to train the imported baseline.")
    splits, groups, hashes, runs = set(), {}, {}, []
    for row in request.rows:
        if row.split not in ("train", "validation", "test") or row.label not in CLASSES or row.origin not in ("new", "existing"):
            raise ValueError("Each recording needs a supported split, class label and data origin.")
        group = row.group.strip()
        if not group: raise ValueError("Group identifiers cannot be blank.")
        if group in groups and groups[group] != row.split: raise ValueError(f"Group {group} crosses data splits. Keep an entire scenario in one split.")
        groups[group] = row.split; splits.add(row.split)
        df = read_recording(row.id, bundle)
        digest = fingerprint(df, bundle["config"]["feature_cols"])
        if digest in hashes: raise ValueError("Duplicate sensor traces were selected. Remove renamed or time-shifted copies.")
        hashes[digest] = row.split
        if row.onset is not None and (row.label == "No leak" or not float(df.TIME.iloc[0]) <= row.onset <= float(df.TIME.iloc[-1])):
            raise ValueError("Verified leak onset must lie inside a leak recording's TIME range.")
        runs.append((df, {**row.model_dump(), "name": read_json(DATA / f"{row.id}.json", {}).get("name", row.id), "fingerprint": digest}))
    if splits != {"train", "validation", "test"}: raise ValueError("Select separate training, validation and test groups.")
    for split in splits:
        labels = {r["label"] == "No leak" for _, r in runs if r["split"] == split}
        if labels != {True, False}: raise ValueError(f"The {split} split needs both a non-leak recording and a leak recording.")
    origins = {r["origin"] for _, r in runs if r["split"] == "train"}
    if origins != {"existing", "new"}: raise ValueError("Training needs representative existing data as well as new data to reduce forgetting.")
    return runs


def job_update(job_id, **changes):
    with LOCK:
        path = DATA / f"job-{job_id}.json"
        job = read_json(path, {}); job.update(changes); job["updated"] = now()
        write_json(path, job)
    return job


def train_worker(job_id, request, runs):
    candidate_id = str(uuid.uuid4()); folder = MODELS / candidate_id
    try:
        job_update(job_id, status="running", phase="Loading a copy of the baseline weights", progress=.02)
        shutil.copytree(MODELS / request.model_id, folder)
        bundle = load_bundle(folder)
        meta = {**bundle["meta"], "id": candidate_id, "name": request.name.strip(), "parent_id": request.model_id, "kind": "fine-tuned", "created": now()}
        bundle["meta"] = meta
        tf = tensorflow()
        for net, loss in [(bundle["detector"], "binary_crossentropy"), (bundle["locator"], "sparse_categorical_crossentropy")]:
            net.compile(optimizer=tf.keras.optimizers.Adam(request.learning_rate, clipnorm=1), loss=loss)
        datasets = {}
        for split in ("train", "validation"):
            xs, ys, eligible_masks = [], [], []
            for df, row in runs:
                if row["split"] != split: continue
                x, labels, eligible = training_windows(df, bundle, row)
                xs.append(x); ys.append(labels); eligible_masks.append(eligible)
            datasets[split] = (np.concatenate(xs), np.concatenate(ys), np.concatenate(eligible_masks))
        x, y, eligible = datasets["train"]; vx, vy, validation_eligible = datasets["validation"]
        locator_indices = np.array([meta["classes"].index(c) if c in meta["classes"] else -1 for c in CLASSES], dtype=np.int32)
        for split in ("train", "validation"):
            _, labels, locator_eligible = datasets[split]
            if len(np.unique(labels > 0)) != 2:
                raise ValueError(f"The {split} split has no usable leak or non-leak windows. Check onset timestamps and recording lengths.")
            if not locator_eligible.any():
                raise ValueError(f"The {split} split has no usable locator clips in the first six clips of its leak recordings. Check onset timestamps; V3 expects recordings that start near the event.")
        vloc = locator_indices[vy]
        validation_mask = validation_eligible & (vloc >= 0)
        best, stale, history = float("inf"), 0, []
        best_weights = None
        rng = np.random.default_rng(42)
        for epoch in range(request.epochs):
            order = rng.permutation(len(x))
            for batch, start in enumerate(range(0, len(x), 64)):
                idx = order[start:start+64]
                bundle["detector"].train_on_batch(x[idx], (y[idx] > 0).astype(np.float32))
                loc_mask = eligible[idx] & (locator_indices[y[idx]] >= 0)
                if loc_mask.any():
                    bundle["locator"].train_on_batch(x[idx][loc_mask], locator_indices[y[idx]][loc_mask])
                job_update(job_id, phase=f"Updating existing weights · epoch {epoch+1}/{request.epochs}", progress=.05 + .7*(epoch+(start+len(idx))/len(x))/request.epochs, epoch=epoch+1)
            dp = np.clip(scores_for(bundle["detector"], vx).reshape(-1), 1e-7, 1-1e-7)
            lp = np.clip(scores_for(bundle["locator"], vx[validation_mask]), 1e-7, 1)
            truth = (vy > 0).astype(float)
            loss = float(np.mean(-truth*np.log(dp)-(1-truth)*np.log(1-dp)) - np.mean(np.log(lp[np.arange(len(lp)), vloc[validation_mask]])))
            history.append({"epoch": epoch+1, "validation_loss": loss})
            if loss < best:
                best=loss; stale=0
                best_weights=[bundle["detector"].get_weights(), bundle["locator"].get_weights()]
            else: stale += 1
            if stale >= 3: break
        bundle["detector"].set_weights(best_weights[0]); bundle["locator"].set_weights(best_weights[1])
        job_update(job_id, phase="Comparing both versions on the same untouched test groups", progress=.82)
        tests = [(df, r) for df, r in runs if r["split"] == "test"]
        base_metrics = evaluate_runs(get_bundle(request.model_id), tests)
        candidate_metrics = evaluate_runs(bundle, tests)
        meta["evaluation"] = {"baseline": base_metrics, "candidate": candidate_metrics, "scope": "User-selected independent test recordings. Delay covers detected runs with supplied verified onset only; no onset means whole-scenario labels.", "test_groups": sorted({r["group"] for _, r in tests})}
        meta["training"] = {"epochs_completed": len(history), "learning_rate": request.learning_rate, "history": history, "scaler_refitted": False, "locator_training_clip_limit": 6 if "No leak" not in meta["classes"] else None, "locator_clip_selection": "First six recording clips, with non-leak/onset filtering" if "No leak" not in meta["classes"] else "All labeled clips", "recordings": [r for _, r in runs]}
        bundle["detector"].save(folder / "lstm_leak_classifier.keras")
        bundle["locator"].save(folder / "lstm_leak_location.keras")
        write_json(folder / "manifest.json", meta)
        add_version(meta, bundle)
        job_update(job_id, status="complete", phase="Candidate ready for review; baseline remains active", progress=1, model_id=candidate_id, evaluation=meta["evaluation"], history=history)
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        job_update(job_id, status="failed", phase="Training failed", error=str(exc))


@app.post("/api/training")
def train(request: TrainingRequest):
    bundle = get_bundle(request.model_id)
    runs = validate_training(request, bundle)
    if sum(len(df) for df, _ in runs) > 200000: raise ValueError("Local training limit is 200,000 total sensor rows per job.")
    with LOCK:
        if any(read_json(p, {}).get("status") in ("queued", "running") for p in DATA.glob("job-*.json")):
            raise HTTPException(409, "A training job is already running.")
        job_id = str(uuid.uuid4())
        job = job_update(job_id, id=job_id, status="queued", phase="Waiting for training worker", progress=0, baseline_id=request.model_id, created=now())
    threading.Thread(target=train_worker, args=(job_id, request, runs), daemon=True).start()
    return job


@app.get("/api/training/{job_id}")
def training_status(job_id: str):
    path = DATA / f"job-{ident(job_id)}.json"
    if not path.exists(): raise HTTPException(404, "Training job not found.")
    return read_json(path, {})


@app.get("/api/training")
def recent_training():
    with LOCK:
        jobs = [read_json(p, {}) for p in DATA.glob("job-*.json")]
    return sorted(jobs, key=lambda j: j.get("created", ""), reverse=True)[:20]


# An interrupted process cannot honestly claim its former job is still running.
for old_job in DATA.glob("job-*.json"):
    old = read_json(old_job, {})
    if old.get("status") in ("queued", "running"):
        job_update(old["id"], status="failed", phase="Interrupted by server restart", error="Start a new job; the baseline weights were preserved.")
