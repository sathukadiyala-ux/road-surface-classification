"""Runs without PyTorch: shows how the automaton cleans a noisy prediction stream
and writes outputs/fsm_demo.png and outputs/fsm_diagram.md."""
import random
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import CLASSES, OUT_DIR, CONFIRM_FRAMES
from fsm import RoadStateFSM, mermaid_diagram

random.seed(1)
truth = ["dry"] * 40 + ["wet"] * 40 + ["damaged"] * 30 + ["dry"] * 30
noisy = [t if random.random() > 0.25 else random.choice(CLASSES + ["unk"]) for t in truth]

fsm = RoadStateFSM(CLASSES, CONFIRM_FRAMES)
states = fsm.run(noisy)
idx = {c: i for i, c in enumerate(["INIT"] + CLASSES)}

OUT_DIR.mkdir(exist_ok=True)
fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
for a, data, title in zip(ax, [truth, noisy, states], ["Ground truth", "Noisy CNN output (25% errors)", f"Automaton output (K={CONFIRM_FRAMES})"]):
    a.step(range(len(data)), [idx.get(d, -1) for d in data], where="mid")
    a.set_yticks(range(len(idx))); a.set_yticklabels(idx.keys()); a.set_title(title, fontsize=10)
plt.tight_layout(); plt.savefig(OUT_DIR / "fsm_demo.png", dpi=150)

flick = lambda s: sum(1 for a, b in zip(s, s[1:]) if a != b)
print(f"State changes  noisy CNN: {flick(noisy)}   automaton: {flick(states)}   truth: {flick(truth)}")
acc = lambda p: sum(a == b for a, b in zip(p, truth)) / len(truth)
print(f"Accuracy       noisy CNN: {acc(noisy):.2%}  automaton: {acc(states):.2%}")
open(OUT_DIR / "fsm_diagram.md", "w").write("```mermaid\n" + mermaid_diagram(CLASSES) + "\n```\n")
print(fsm.transition_table())
