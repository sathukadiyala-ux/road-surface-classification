"""
Explainability tools.

gradcam():          Grad-CAM heatmap showing which image regions the CNN used for a class.
segment_damage():   approximate damage mask (potholes + cracks). This is a classical image-processing
                    heuristic (dark-region and thin-dark-line filters) restricted to the Grad-CAM area.
                    It is NOT a trained segmentation model. For a true model, train a U-Net on pixel masks.
"""
import base64
import cv2
import numpy as np

DAMAGE_CLASS = "damaged"


def gradcam(model, x, class_idx=None):
    """x: (1,3,H,W) tensor. Returns (probs list, cam HxW float in [0,1], class index used)."""
    import torch
    import torch.nn.functional as F
    store = {}
    layer = model.features[-1]

    def fwd(_, __, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    h = layer.register_forward_hook(fwd)
    try:
        model.zero_grad()
        with torch.enable_grad():
            out = model(x)
            probs = torch.softmax(out, 1)[0].detach().cpu().tolist()
            idx = int(out.argmax(1)) if class_idx is None else int(class_idx)
            out[0, idx].backward()
    finally:
        h.remove()
    w = store["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * store["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return probs, cam, idx


def segment_damage(rgb, cam=None, cam_thr=0.35):
    """rgb: HxWx3 uint8. Returns (bool mask, damaged-area percent)."""
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    dark = gray.astype(np.int16) < bg.astype(np.int16) - 25                      # pothole-like dark areas
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    cand = dark | (blackhat > 18)                                                  # thin dark lines (cracks)
    if cam is not None:
        region = cv2.dilate((cam > cam_thr).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        cand &= region
    m = cand.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 30:
            keep[lab == i] = 1
    return keep.astype(bool), 100.0 * keep.sum() / keep.size


def overlay_cam(rgb, cam, alpha=0.45):
    heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, heat, alpha, 0)


def overlay_mask(rgb, mask, alpha=0.5):
    out = rgb.copy()
    red = np.zeros_like(rgb); red[..., 0] = 255
    out[mask] = cv2.addWeighted(rgb, 1 - alpha, red, alpha, 0)[mask]
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 220, 0), 1)
    return out


def to_data_uri(rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
