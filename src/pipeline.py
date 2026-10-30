"""
End-to-end system: camera/video/image-folder -> CNN -> finite automaton -> stable state + warning.

Usage:
    python src/pipeline.py --source video.mp4            # video file
    python src/pipeline.py --source 0                    # webcam
    python src/pipeline.py --source data/split/test/wet  # folder of images (treated as a frame sequence)
    add --save outputs/demo.mp4 to write the annotated video, --no-show for headless runs.
"""
import argparse, csv
from pathlib import Path
import cv2, torch
from PIL import Image
from config import *
from data import eval_tf
from model import build_model
from fsm import RoadStateFSM, UNKNOWN

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    ck = torch.load(MODEL_DIR / "best.pt", map_location=dev)
    m = build_model(len(ck["classes"]), pretrained=False).to(dev)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck["classes"]


def frames_from(source):
    p = Path(source)
    if p.is_dir():
        for f in sorted(p.glob("*")):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                yield cv2.imread(str(f))
        return
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    while True:
        ok, fr = cap.read()
        if not ok: break
        yield fr
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--save", default=None)
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()

    model, classes = load_model()
    fsm = RoadStateFSM(classes, CONFIRM_FRAMES)
    tf = eval_tf()
    writer, rows = None, []

    for i, frame in enumerate(frames_from(a.source)):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            prob = torch.softmax(model(tf(Image.fromarray(rgb)).unsqueeze(0).to(dev)), 1)[0]
        conf, idx = prob.max(0)
        raw = classes[idx.item()]
        symbol = raw if conf.item() >= CONFIDENCE_THRESHOLD else UNKNOWN
        stable = fsm.step(symbol)
        rows.append([i, raw, f"{conf.item():.3f}", symbol, stable])

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
        cv2.putText(frame, f"CNN: {raw} ({conf.item():.2f})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, f"FSM state: {stable.upper()}  limit: {fsm.speed_limit}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if a.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            writer.write(frame)
        if not a.no_show:
            cv2.imshow("Road surface monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break

    if writer: writer.release()
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "pipeline_log.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "cnn_raw", "confidence", "fsm_input", "fsm_state"]); w.writerows(rows)
    changes_raw = sum(1 for p, q in zip(rows, rows[1:]) if p[1] != q[1])
    changes_fsm = sum(1 for p, q in zip(rows, rows[1:]) if p[4] != q[4])
    print(f"Frames: {len(rows)} | raw prediction changes: {changes_raw} | FSM state changes: {changes_fsm}")
    print("Log saved to outputs/pipeline_log.csv")


if __name__ == "__main__":
    main()
