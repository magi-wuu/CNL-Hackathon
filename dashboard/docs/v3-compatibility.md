# LeakDetectionV3 compatibility review

Reference: `reference/LeakDetectionV3.ipynb`, supplied on 2026-10-04. Notebook SHA-256: `5eb07a86243bc7435107b2efdba6b6ab4ef1ed48a43e2b122b3c6bc5d92f10c6`. The reference is preserved unchanged; its code and saved outputs were read, but it was not executed or retrained.

## Verified code contract

- Binary detector: Sequential LSTM 64 → dropout → LSTM 32 → Dense 32 → dropout → one sigmoid output.
- Locator: same hidden structure, seven softmax outputs in `LOCA, LOCAC, SGATR, SGBTR, SLBIC, FLB, LLB` order. No `No leak` output.
- Features: sorted numeric intersection across operational CSVs, excluding TIME and the listed direct-leak/derived gauges. Use the exported config's exact ordered list, rather than hard-coding a new list. The installed package has 81 features; V3 derives its count from data.
- Float32 sensor values, saved StandardScaler, 12 samples/clip, stride 5, 10-second sampling.
- Detector alarm: score ≥0.7 for three consecutive clips. The 0.5 cutoff used by V3's clip classification report is a separate metric, not its alarm policy.
- Locator training and validation: first six clips of each leak recording (`clip_no < N_EARLY`, N_EARLY=6), not six clips after a verified onset.
- Location vote: per-clip argmax over the first three recording clips, then `Series.mode()[0]`. A three-way tie is resolved by alphabetical string ordering, rather than output-index order or mean score.

The user confirmed that the four already installed model files are the current exports for this V3 notebook. The dashboard keeps those weights and aligns these preprocessing and voting rules. It invokes the seven-class locator only after an alarm and reveals the vote when the alarm and all three clips are available. The notebook evaluates the locator separately on known leak recordings; it does not implement a combined deployed inference service. Alarm gating comes from the user's confirmed integration requirement.

## Deliberate dashboard evaluation safeguards

- Window timestamps use the actual CSV end-reading TIME. With TIME=0, T=12 and a 10-second interval, the first window ends at 110 seconds. V3's alarm report uses `(clip_index * STRIDE + T) * SAMPLE_SEC`, which reports 120 seconds for the same window (a 10-second offset). V3's plotting cell instead uses `TIME[T-1::STRIDE]`, agreeing with the dashboard. Do not copy the report's arithmetic into playback timestamps.
- V3 labels whole leak recordings as leaks and calls alarm time “after the leak started” without reading onset metadata. The dashboard only reports onset-based delay when a verified onset is supplied.
- V3's locator evaluation counts held-out leak recordings independently of detector success. The dashboard's operational location metric counts detected leak recordings with an available vote; these denominators differ and the percentages are not directly comparable.
- V3 uses grouped recording splits when an event type has multiple recordings, but time-splits already overlapping windows for singleton types. The dashboard keeps training/validation/test groups separate and does not adopt that singleton fallback.
- Fine-tuning starts from saved weights, keeps the saved scaler, and uses independent groups. For a seven-output locator it restricts updates and validation to each leak recording's first six clips, excluding verified pre-onset windows. The detector still uses all clips. If no eligible location clips remain, the job fails with an explanation rather than using late-event clips silently.

## Notebook execution issues found in source

Cell numbers below are zero-based notebook indices:

1. Cell 2 begins with `wrong = r[r.real != r.predicted]` and `loc2.save(...)` before defining `r` or `loc2`. A clean top-to-bottom run would fail here. The later training and final save in the same cell are correctly ordered once that stale leading block is removed.
2. Cell 3 uses `rec` as a recording-name string in its loop. Cell 7 later treats `rec` as a DataFrame with `rec["clip_no"] = ...`; no such DataFrame is defined in this notebook. Cell 2 already contains the current first-three-clip evaluation, so cell 7 appears to be a leftover older evaluation block.
3. Setup/training use `/content/NPPAD/Operation_csv_data/`; later replay/list/copy cells use `/content/NuclearPowerPlantAccidentData/Operation_csv_data/`. The setup does not create the latter directory, so those cells require a separate existing Colab state or a path correction.

Saved output exists but every execution_count is null. It cannot establish a clean execution order, identify the latest exported weights, or resolve these source errors. The dashboard does not execute attached notebook cells as instructions.
