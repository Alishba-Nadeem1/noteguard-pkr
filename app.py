from __future__ import annotations
import io
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps
from torchvision import models
# ────────────────────────────────────────────────────────────────────
#  Config
# ────────────────────────────────────────────────────────────────────
IMG_SIZE = 224
MODEL_PATH = Path("model.pth")
DEFAULT_THRESHOLD = 0.50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.set_page_config(
    page_title="PKR Note Authenticator",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ────────────────────────────────────────────────────────────────────
#  Custom CSS — dark theme, animated gradient bg, glass cards, glow
# ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@500&display=swap');
    .stApp {
        background: radial-gradient(1200px 800px at 10% -10%, #1b2a4e 0%, transparent 60%),
                    radial-gradient(1000px 700px at 110% 10%, #4a1d5c 0%, transparent 55%),
                    linear-gradient(135deg, #060814 0%, #0a0f24 50%, #0b0620 100%);
        background-attachment: fixed;
        color: #e6ecff;
        font-family: 'Space Grotesk', sans-serif;
    }
    /* Animated aurora blob */
    .stApp::before {
        content: "";
        position: fixed; inset: 0;
        background: conic-gradient(from 180deg at 50% 50%,
            rgba(99,102,241,.15), rgba(236,72,153,.12),
            rgba(34,211,238,.15), rgba(99,102,241,.15));
        filter: blur(80px); opacity: .35;
        animation: spin 30s linear infinite;
        pointer-events: none; z-index: 0;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    section.main > div { position: relative; z-index: 1; }
    /* Hero title */
    .hero {
        text-align: center;
        padding: 2.2rem 1rem 1.6rem;
    }
    .hero h1 {
        font-size: clamp(2.2rem, 5vw, 3.6rem);
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #7dd3fc, #c4b5fd, #f9a8d4);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    .hero p {
        color: #9aa4c7; margin-top: .4rem;
        font-size: 1.05rem;
    }
    .badge {
        display:inline-block; padding: 4px 12px; border-radius: 999px;
        background: rgba(125,211,252,.1); color:#7dd3fc;
        border: 1px solid rgba(125,211,252,.3);
        font-family: 'JetBrains Mono', monospace; font-size: .75rem;
        margin-bottom: .8rem; letter-spacing: .08em;
    }
    /* Glass cards */
    .glass {
        background: rgba(17, 22, 46, 0.55);
        backdrop-filter: blur(18px) saturate(140%);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 20px 60px -20px rgba(0,0,0,.6);
    }
    /* Verdict cards */
    .verdict {
        border-radius: 24px; padding: 2rem 1.5rem; text-align:center;
        position: relative; overflow: hidden;
        border: 1px solid rgba(255,255,255,.08);
    }
    .verdict h2 { font-size: 2.2rem; margin: .5rem 0; letter-spacing:-0.02em; }
    .verdict.real {
        background: linear-gradient(135deg, rgba(34,197,94,.18), rgba(16,185,129,.08));
        box-shadow: 0 0 60px -10px rgba(34,197,94,.4), inset 0 0 40px rgba(34,197,94,.05);
    }
    .verdict.real h2 { color: #4ade80; text-shadow: 0 0 30px rgba(74,222,128,.5); }
    .verdict.fake {
        background: linear-gradient(135deg, rgba(239,68,68,.18), rgba(236,72,153,.08));
        box-shadow: 0 0 60px -10px rgba(239,68,68,.4), inset 0 0 40px rgba(239,68,68,.05);
    }
    .verdict.fake h2 { color: #f87171; text-shadow: 0 0 30px rgba(248,113,113,.5); }
    .verdict .sub { color: #9aa4c7; font-family: 'JetBrains Mono', monospace; font-size:.85rem; }
    /* Confidence meter */
    .meter-wrap { margin-top: 1.2rem; }
    .meter-label {
        display:flex; justify-content:space-between;
        font-family:'JetBrains Mono',monospace; font-size:.8rem;
        color:#9aa4c7; margin-bottom:.4rem;
    }
    .meter {
        height: 14px; border-radius: 999px;
        background: rgba(255,255,255,.06);
        overflow: hidden; position:relative;
        border: 1px solid rgba(255,255,255,.05);
    }
    .meter-fill {
        height:100%; border-radius:999px;
        background: linear-gradient(90deg, #22d3ee, #a78bfa, #f472b6);
        box-shadow: 0 0 20px rgba(167,139,250,.7);
        transition: width 1.2s cubic-bezier(.22,1,.36,1);
    }
    /* Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(17,22,46,.5);
        border: 2px dashed rgba(167,139,250,.35);
        border-radius: 20px; padding: 1rem;
        transition: all .3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(167,139,250,.7);
        box-shadow: 0 0 40px -10px rgba(167,139,250,.5);
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(6,8,20,.7) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,.06);
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg,#6366f1,#a78bfa);
        color:white; border:none; border-radius:12px;
        padding:.6rem 1.4rem; font-weight:600;
        box-shadow: 0 10px 30px -10px rgba(99,102,241,.6);
        transition: transform .2s ease, box-shadow .2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px -10px rgba(167,139,250,.8);
    }
    /* Image frame */
    [data-testid="stImage"] img {
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,.08);
        box-shadow: 0 20px 50px -20px rgba(0,0,0,.7);
    }
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)
# ────────────────────────────────────────────────────────────────────
#  Model
# ────────────────────────────────────────────────────────────────────
@dataclass
class Prediction:
    label: str
    real_prob: float
    fake_prob: float
    is_real: bool
@st.cache_resource(show_spinner="Loading model…")
def load_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if MODEL_PATH.exists():
        try:
            state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        except TypeError:
            state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
    else:
        st.warning("model.pth not found — running with random weights (demo mode).")
    model.to(DEVICE).eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE))
    return model
@st.cache_resource
def get_transform():
    # FIXED: must match training normalization exactly (train.py / train_v3.py
    # used Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]), NOT ImageNet stats.
    # Using the wrong normalization here was flipping predictions completely
    # (a note that tested 99.98% fake with correct normalization was showing
    # 99.9% "Authentic" with ImageNet normalization).
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
def predict(model: nn.Module, img: Image.Image, threshold: float) -> Prediction:
    tf = get_transform()
    x = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    fake_prob, real_prob = float(probs[0]), float(probs[1])
    is_real = real_prob >= threshold
    return Prediction(
        label="Authentic" if is_real else "Suspected Counterfeit",
        real_prob=real_prob, fake_prob=fake_prob, is_real=is_real,
    )
def gradcam(model: nn.Module, img: Image.Image, target_class: int) -> np.ndarray:
    # FIXED: takes target_class as a parameter now, so the heatmap always
    # matches the class actually being displayed (threshold-based verdict),
    # instead of silently using argmax which could show a different class
    # than the one in the verdict card when threshold != 0.5.
    tf = get_transform()
    x = tf(img).unsqueeze(0).to(DEVICE)
    x.requires_grad_(True)
    feats, grads = [], []
    target_layer = model.layer4[-1]
    h1 = target_layer.register_forward_hook(lambda _, __, o: feats.append(o))
    h2 = target_layer.register_full_backward_hook(lambda _, __, go: grads.append(go[0]))
    try:
        logits = model(x)
        model.zero_grad(set_to_none=True)
        logits[0, target_class].backward()
        g = grads[0][0].mean(dim=(1, 2))
        f = feats[0][0]
        cam = F.relu((g[:, None, None] * f).sum(dim=0)).detach().cpu().numpy()
    finally:
        h1.remove(); h2.remove()
    cam = cv2.resize(cam, (img.size[0], img.size[1]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    base = np.array(img.convert("RGB"))
    return cv2.addWeighted(base, 0.55, heat, 0.45, 0)
# ────────────────────────────────────────────────────────────────────
#  UI
# ────────────────────────────────────────────────────────────────────
def main() -> None:
    with st.sidebar:
        st.markdown("### Controls")
        threshold = st.slider(
            "Decision threshold (Real ≥)", 0.10, 0.90, DEFAULT_THRESHOLD, 0.01,
            help="Higher = stricter about calling a note real.",
        )
        st.markdown("---")
        st.markdown("### Model")
        st.markdown(
            f"- Backbone: **ResNet18**\n"
            f"- Input: **{IMG_SIZE}×{IMG_SIZE}**\n"
            f"- Device: **{DEVICE.type.upper()}**"
        )
        st.markdown("---")
        st.caption(
            "Note: Student/portfolio project. Trained on synthetic forgery patterns. "
            "Do not use as a real financial authentication tool."
        )
    st.markdown(
        """
        <div class="hero">
            <div class="badge">AI · GRAD-CAM · RESNET-18</div>
            <h1>PKR Note Authenticator</h1>
            <p>Upload a Pakistani currency note — get instant authenticity analysis with a visual heatmap.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    model = load_model()
    uploaded = st.file_uploader(
        "Drop a note image here (JPG · PNG · WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if uploaded is None:
        st.markdown(
            """
            <div class="glass" style="text-align:center; margin-top:1rem;">
                <h3 style="margin:0; color:#c4b5fd;">Ready when you are</h3>
                <p style="color:#9aa4c7; margin-top:.4rem;">Drag a note above to run authentication.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    try:
        img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        st.error(f"Could not read image: {e}")
        return
    with st.spinner("Analyzing note…"):
        pred = predict(model, img, threshold)
        # FIXED: pass the class matching the displayed verdict (0=fake, 1=real)
        target_class = 1 if pred.is_real else 0
        try:
            cam_img = gradcam(model, img, target_class)
        except Exception as e:
            cam_img = None
            st.warning(f"Grad-CAM unavailable: {e}")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("#### Uploaded note")
        st.image(img, use_container_width=True)
        if cam_img is not None:
            st.markdown("#### Grad-CAM focus")
            st.image(cam_img, use_container_width=True,
                     caption="Hot regions = pixels that drove the decision")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        cls = "real" if pred.is_real else "fake"
        icon = "✓" if pred.is_real else "✕"
        conf = pred.real_prob if pred.is_real else pred.fake_prob
        st.markdown(
            f"""
            <div class="verdict {cls}">
                <div style="font-size:3rem;">{icon}</div>
                <h2>{pred.label}</h2>
                <div class="sub">threshold · {threshold:.2f}   |   device · {DEVICE.type.upper()}</div>
                <div class="meter-wrap">
                    <div class="meter-label">
                        <span>confidence</span><span>{conf*100:.1f}%</span>
                    </div>
                    <div class="meter"><div class="meter-fill" style="width:{conf*100:.1f}%"></div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="glass" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown("#### Probability breakdown")
        c1, c2 = st.columns(2)
        c1.metric("Real",  f"{pred.real_prob*100:.1f}%")
        c2.metric("Fake",  f"{pred.fake_prob*100:.1f}%")
        st.markdown(
            f"""
            <div class="meter-wrap">
                <div class="meter-label"><span>real</span><span>{pred.real_prob*100:.1f}%</span></div>
                <div class="meter"><div class="meter-fill" style="width:{pred.real_prob*100:.1f}%; background:linear-gradient(90deg,#22c55e,#10b981);"></div></div>
            </div>
            <div class="meter-wrap">
                <div class="meter-label"><span>fake</span><span>{pred.fake_prob*100:.1f}%</span></div>
                <div class="meter"><div class="meter-fill" style="width:{pred.fake_prob*100:.1f}%; background:linear-gradient(90deg,#ef4444,#ec4899);"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
if __name__ == "__main__":
    main()