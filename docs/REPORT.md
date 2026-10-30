# Road Surface Condition Classification using Deep Learning with a Finite Automata Based State Monitoring Model

## Abstract
Road surface condition strongly affects vehicle safety, yet camera-based classifiers often give unstable,
flickering predictions from frame to frame. This project classifies road surfaces (dry, wet, snow, gravel,
damaged) using a MobileNetV2 convolutional neural network trained with transfer learning and fine-tuning.
The per-frame predictions are fed as input symbols to a finite automaton that confirms a change of road
state only after K consecutive agreeing frames and ignores low-confidence frames. The automaton outputs a
stable road state with an associated driver warning and recommended speed. The combination joins
deep learning (perception) with automata theory (reliable decision logic).

## 1. Problem statement
Given a stream of road images, determine the current road surface condition reliably enough to trigger
warnings, without false alarms caused by single-frame misclassifications.

## 2. Methodology
1. **Data:** images organised by class, split 70/15/15 into train/val/test. Augmentation: random resized crop,
   flip, colour jitter, rotation.
2. **Model:** MobileNetV2 pretrained on ImageNet; classifier head replaced. Stage 1 trains only the head,
   stage 2 fine-tunes all layers at a lower learning rate.
3. **Automaton layer:** see Section 3.
4. **System:** `pipeline.py` runs frame -> CNN -> softmax -> threshold -> automaton -> overlay and log.

## 3. Finite automaton (formal definition)
A = (Q, Sigma, delta, q0, Lambda, lambda)

- **Sigma** = {dry, wet, snow, gravel, damaged, unk}. `unk` is emitted when CNN confidence < threshold.
- **Q** = (current, candidate, count), current in {INIT, dry, wet, snow, gravel, damaged},
  candidate in {none, dry, ...}, count in {0..K-1}. Finite because count is bounded by K.
- **q0** = (INIT, none, 0)
- **delta:**
  - `unk` -> no change
  - symbol = current -> candidate cleared, count = 0
  - symbol != current -> if symbol = candidate then count + 1 else candidate = symbol, count = 1;
    when count = K the current state becomes the symbol
- **Lambda** = warning messages and speed limits; **lambda**(q) = message of q.current (Moore output)

### State diagram (stable-state level, each edge = K consecutive frames of that class)
Generated into `outputs/fsm_diagram.md` by `python src/fsm_demo.py` (Mermaid format; paste into
https://mermaid.live or a Markdown viewer that supports Mermaid to export an image).

### Example trace (K = 3)
| frame | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| CNN | dry | dry | dry | wet | dry | wet | wet | unk | wet |
| state | INIT | INIT | dry | dry | dry | dry | dry | dry | wet |

The isolated `wet` at frame 4 is rejected; the real change to wet is accepted at frame 9.

## 4. Results (fill in after training)
| Metric | Value |
|---|---|
| Test accuracy | |
| Macro F1 | |
| State changes, raw CNN | |
| State changes, with automaton | |
Include `outputs/confusion_matrix.png`, `outputs/classification_report.txt`, and `outputs/fsm_demo.png`.

## 5. Limitations and future work
- Accuracy depends on dataset quality and diversity (lighting, camera angle, regions).
- The automaton adds latency of K frames when the road truly changes.
- Future work: deploy on Raspberry Pi/Jetson (edge AI), add GPS/speed fusion, use a probabilistic
  model (HMM) in place of the deterministic automaton, and compare with LSTM temporal models.
