# Road Surface Condition Classification using Deep Learning with a Finite Automata State Monitor

One project that combines three ideas:

1. **Road surface condition classification** (dry / wet / snow / gravel / damaged)
2. **Deep learning** (MobileNetV2 CNN, transfer learning + fine-tuning)
3. **Finite automata** (a Moore-machine style automaton that turns noisy per-frame CNN predictions into a stable road state with warnings)

```
camera frame -> CNN (class + confidence) -> input symbol -> Finite Automaton -> stable state + warning
```

Suggested hackathon track: **Track 01 - AI, Generative AI & Intelligent Agents** (Computer Vision).

## Project structure
```
src/config.py           settings (classes, epochs, automaton K, confidence threshold)
src/prepare_dataset.py  split data/raw into train/val/test (or generate a synthetic demo set)
src/data.py             dataloaders and augmentation
src/model.py            MobileNetV2 transfer-learning model
src/train.py            two-stage training (head, then fine-tune)
src/evaluate.py         test accuracy, classification report, confusion matrix
src/fsm.py              the finite automaton (formal definition in the docstring)
src/pipeline.py         full system: video/webcam/image folder -> CNN -> automaton -> overlay + CSV log
src/webapp.py           dashboard: upload road images -> CNN + automaton result, charts, CSV export
src/map_dataset.py      (optional) copy a downloaded dataset into data/raw by keywords
src/explain.py          Grad-CAM + heuristic damage segmentation
src/fsm_demo.py         runs without PyTorch; visualises noise filtering by the automaton
sample_data/            synthetic damaged-road test images (not real data)
tests/test_fsm.py       unit tests for the automaton
docs/REPORT.md          abstract, methodology, automaton formal definition, results template
```

## Setup
```
pip install -r requirements.txt
```

## Quick start
1. **Add data.** Put images in `data/raw/<class>/` with folder names matching `CLASSES` in `src/config.py`
   (dry, wet, snow, gravel, damaged). Suggested public sources: RSCD (Road Surface Classification Dataset),
   RoadSaW, or Kaggle road damage / pothole datasets. Check each dataset's license and class mapping yourself.
   No dataset yet? `python src/prepare_dataset.py --synthetic` makes a toy set so you can test the pipeline.
   Synthetic data only proves the code runs; it gives no meaningful results.
2. `python src/prepare_dataset.py` (split 70/15/15)
3. `python src/train.py` (saves `models/best.pt`)
4. `python src/evaluate.py` (writes `outputs/confusion_matrix.png`, `classification_report.txt`)
5. `python src/pipeline.py --source your_video.mp4 --save outputs/demo.mp4`
   (or `--source 0` for webcam, or a folder of images; add `--no-show` on a headless machine)

## Run the automaton on its own (no PyTorch needed)
```
python tests/test_fsm.py
python src/fsm_demo.py
```
The demo shows a noisy stream where the CNN output flips state 33 times; the automaton reduces that to 4
(the true sequence has 3 changes). The extra change comes from the INIT start state. Note that frame-level
accuracy against ground truth can go slightly down, because the automaton intentionally waits K frames before
switching, so it lags at real transitions. What you gain is stability. Report both numbers honestly.

## Tuning
- `CONFIRM_FRAMES` (K): larger means more stable but slower to react. Try 3 to 10 depending on frame rate.
- `CONFIDENCE_THRESHOLD`: predictions below it become the `unk` symbol and are ignored by the automaton.

## Status of this code
The automaton and its tests were run and pass. The deep learning scripts (train, evaluate, pipeline) were
written but not executed in the environment where this zip was produced (no PyTorch available there),
so run them once on your machine and fix any small path or version issues. Training needs real data for real results.

## Web dashboard
After training (models/best.pt exists): `python src/webapp.py`, then open http://127.0.0.1:5000.
Features: drag-and-drop upload, KPI cards (state, speed limit, state changes raw vs automaton, confidence),
state timeline and confidence charts, live K and confidence sliders, per-frame table with thumbnails,
click-a-row Grad-CAM heatmap and damage segmentation (src/explain.py), CSV export, print report, light/dark theme, and the test-set evaluation report (if you ran evaluate.py).

Note: damage segmentation is a classical image-processing estimate restricted to the Grad-CAM area, not a trained segmentation model.
For true segmentation, label pixel masks and train a U-Net.
