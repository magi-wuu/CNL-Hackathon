# Dashboard presentation walkthrough

Duration: five minutes, with questions afterward. Aim to finish at 4:40. Audience: hackathon judges with mixed technical backgrounds. This plan covers the dashboard and leak-detection workflow; other team work needs its own time and evidence.

## Main recommendation

Present one operator scenario, rather than touring every widget: inspect a recording, observe a detector alarm, locate the suspected component, review evidence, then explain model maintenance. Use a rehearsed recorded-data example with the installed trained model. Keep the explicitly synthetic Demo as a fallback, and disclose it if used.

Credit the supplied LSTM architecture and dataset. Explain the team’s documented modeling changes separately from dashboard integration and visualization. Do not claim to have invented the inherited architecture. The simplified PWR diagram communicates location classes; it does not establish validation on a specific reactor design or an SMR.

## Timed run of show

| Time | Display / action | Point to explain |
| --- | --- | --- |
| 0:00–0:30 | Monitor, paused before the selected recording’s alarm | The engineering problem is interpreting a suspected leak and deciding where to investigate. The project combines leak detection with an interface for inspecting the evidence. State that this presentation replays recorded simulation data; there is no live plant connection. |
| 0:30–1:00 | Selected model name, score panel, sensor trends | The supplied model is loaded once and persists. Each input combines 12 readings of 81 sensors. Readings are 10 seconds apart; a new assessment is available every 50 seconds. Briefly credit the modeling foundation and identify the team’s integration work. |
| 1:00–2:15 | Play at 20×; point to the score, confirmation indicators, highlighted part, event log. Pause at the alarm. | A score is a model output, not a calibrated certainty. One threshold crossing is insufficient: the configured alarm requires ≥0.7 for three consecutive clips. The 3D schematic and event log follow the same recorded timeline. The red area is the model-suspected component category. |
| 2:15–3:00 | Acknowledge, then open Evidence | Acknowledgement records awareness and leaves the alarm latched. The seven-output locator runs after alarm confirmation and votes on the recording’s first three clips. Evidence displays those votes, their times, and mean location scores. The detector answers whether a leak is indicated; the locator ranks supported leak locations. |
| 3:00–3:45 | Model lab; point to the selected version, Update model, and Improve current model | Trained packages can be imported without retraining. Optional labeled data continues training from existing weights, producing a separate candidate. The baseline and scaler are preserved; candidate evaluation uses the same independent test recordings. Engineers choose whether to activate the candidate and can return to an earlier version. |
| 3:45–4:20 | One prepared results slide or a verified saved evaluation; otherwise return to Monitor | Present only verified counts for detected/missed leak recordings and false alarms, plus location accuracy with its denominator. Include onset-based delay only if onset is verified. A replay demonstrates the workflow; it does not establish general accuracy or early small-leak performance. If no audited evaluation is ready, state that limitation instead of displaying old notebook percentages. |
| 4:20–4:40 | Monitor | Close with the demonstrated result: an integrated workflow that connects saved models and recorded sensor data to alarms, suspected locations and inspectable evidence. The next milestone is testing on independent scenarios, emphasizing small-leak detection timing and false alarms. |
| 4:40–5:00 | Stay on Monitor | Timing margin. Stop speaking rather than filling the remaining seconds with extra features. |

The actual alarm timestamp depends on the recording. Rehearse the exact file before choosing the start position; do not assume the synthetic demo’s timestamp applies to model inference.

## Before presenting

1. Start the app and verify that the local model service is reachable and the intended trained model is selected. Models persist; the current screen initially loads the synthetic demo until a CSV is loaded.
2. Use **Load data** (or **Import → Replay data**) to load a known sensor recording. Confirm the source bar says **RECORDED DATA REPLAY**. Record its original scenario path and label, since files from different folders may share a basename such as `1.csv`.
3. Rehearse a known leak recording and a non-leak recording. Verify the leak example’s actual alarm and location result rather than assuming it will succeed. Keep the non-leak example for questions unless both cases comfortably fit the schedule.
4. Confirm whether the example was held out from training. A training example can demonstrate operation but cannot prove generalization. If holdout status is unknown, describe it as an inference/workflow example.
5. Preload the main recording before the judges arrive. Start paused shortly before an informative portion, set 20×, and keep the original file accessible. Loading/scoring a long recording or searching through folders during the talk adds avoidable delay.
6. Set the browser size for the projector and verify that the reactor, chart, controls and event log fit. Make the cursor visible when pointing. Rotate the reactor once briefly; keep the useful cutaway orientation for the rest of the demo.
7. Record a short backup screen capture and save a screenshot of the alarm/evidence. Rehearse switching to the backup without restarting the presentation.
8. Keep an already prepared evaluation view if you want to show candidate-versus-baseline results. Do not start training during the five-minute talk or imply an empty Model lab contains completed evaluation results.

## Synthetic fallback

The **Demo** button uses scripted readings, leak scores and location values; it does not run the trained model. Say this before using it. From a restart at TIME=0 and 20× speed, its full 15-minute recording lasts 45 seconds of presentation time. In this scripted scenario, the first threshold hit occurs at replay 09:20 and the third-hit alarm at 11:00. These are scenario timestamps, not measured model detection performance.

Use it to show synchronized playback, the reactor highlight, acknowledgement and event seeking. Its location outputs are illustrative and do not represent the installed V3 locator. Explain the V3 pipeline from the saved model contract, rather than describing synthetic ranking values as real inference.

## Technical depth to explain aloud

- **Temporal inputs:** the LSTM reads an ordered clip of sensor measurements, rather than just one reading. The exported scaler and feature order accompany the saved weights.
- **Alarm confirmation:** the three consecutive hits define a confirmation policy intended to filter isolated high scores. A controlled comparison is needed before claiming a particular reduction in false alarms.
- **Source visibility:** the selected model for future replays can differ from the model that scored the current recording. The current recording retains its results until scored with another version.
- **Location interpretation:** the seven-class locator has no No leak output, which is why detection and localization are separate. A highlighted circuit is a suspected class, not a precisely identified pipe segment.
- **Optional fine-tuning:** the detector uses all labeled clips; the V3 locator uses the first six recording clips from leak scenarios, excluding verified pre-onset clips. This assumes recordings begin near the event. Updates keep the original scaler and save separate candidate weights.

Keep detailed layer sizes, training hyperparameters, tie-breaking and notebook execution issues for questions. The main talk needs the data flow and the operator workflow.

## Claims and wording

Use “recorded-data replay with simulated real-time presentation,” not “connected to a live reactor.” When the input is a simulator dataset, avoid calling it real plant data. The interface advances according to recorded time; the backend can precompute causal predictions before playback.

Use “model-suspected component” rather than “the exact failure location.” Do not call the 3D schematic a validated digital twin or a physics simulation.

Use “detector score 0.96” rather than “96% certain there is a leak.” Calibration has not been established.

Use “no alarm indicated for this recording” rather than “the reactor is safe.” Leak classification is not an overall plant safety assessment.

Use “fine-tuning creates a candidate for evaluation” rather than “new data always improves the model.” Updates can worsen detection or false alarms.

The dashboard’s software tests establish behavior of ingestion, model loading, timing, voting and training/version safeguards. They are not model-accuracy measurements.

## Questions to prepare for

**Does it run in real time?** The current implementation replays recorded readings and reveals time-aligned predictions. Live sensor ingestion is a future integration, not a demonstrated feature.

**Did your team build the AI?** Credit the supplied architecture and dataset, then explain the team’s actual modeling adaptations and the operational integration demonstrated here.

**How do you know it detects small leaks early?** Present an audited held-out result only if available. The intended evaluation needs known leak size/severity, verified onset, missed detections and false alarms; a high aggregate accuracy or one successful replay is insufficient.

**Why wait for three scores?** To require sustained evidence rather than one isolated high score. This introduces a delay: the first and third qualifying assessments are 100 recorded seconds apart at the current 50-second assessment cadence. It is a configured tradeoff, not a proven optimal policy.

**Why can the locator be wrong?** It classifies seven supported leak categories and uses the first three clips. Similar responses or clips recorded before the event can mislead it. The ranking and sensor evidence remain visible for review.

**Can users improve or replace it?** Yes: import a compatible saved package, or fine-tune existing weights using labeled data and independent groups. Review a separate candidate before selecting it. The saved model and selected version persist locally.

## Choice of emphasis

The evidence perspective favors recorded-data inference because it demonstrates the real model path. The practical perspective favors the predictable synthetic replay for a reliable fallback. The skeptic would question whether one attractive visualization establishes detection accuracy. The longer-term project goal favors an inspectable, maintainable operator workflow. The best balance is a rehearsed recorded-data example, clear provenance, a disclosed fallback and bounded performance claims.

Do not spend most of the talk orbiting the 3D reactor or reading neural-network equations. Use the graphic to explain the suspected location, then use Evidence to show why the result can be investigated.
