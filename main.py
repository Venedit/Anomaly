
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import scipy.ndimage
import os
import pandas as pd
import matplotlib.pyplot as plt


IMG_SIZE = 256
LATENT_DIM = 128
MODEL_PATH = 'autoencoder_checkpoint.pth'


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256)
        )

        self.flatten = nn.Flatten()

        self.linear_enc = nn.Sequential(
            nn.Linear(256 * 16 * 16, LATENT_DIM),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.linear_dec = nn.Sequential(
            nn.Linear(LATENT_DIM, 256 * 16 * 16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.unflatten = nn.Unflatten(1, (256, 16, 16))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.linear_enc(x)
        x = self.linear_dec(x)
        x = self.unflatten(x)
        return self.decoder(x)

@st.cache_resource
def load_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH):
        st.error(f"Файл {MODEL_PATH} не знайдено")
        return None, None, 0.035

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = Autoencoder().to(device)

    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint['model_state_dict'])
        threshold = float(checkpoint.get('threshold', 0.035))
    else:
        model.load_state_dict(checkpoint)
        threshold = 0.035

    model.eval()
    return model, device, threshold

def analyze_image(image, model, device, threshold):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        recon = model(img_tensor)

    img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    recon_np = recon.squeeze().permute(1, 2, 0).cpu().numpy()

    img_blur = scipy.ndimage.gaussian_filter(img_np, sigma=2)
    diff = np.mean(np.abs(img_blur - recon_np), axis=2)
    heatmap = scipy.ndimage.gaussian_filter(diff, sigma=6)

    score = float(np.max(heatmap))
    is_defect = score > threshold

    return {
        "score": score,
        "is_defect": is_defect,
        "heatmap": heatmap,
        "original": img_np,
        "reconstruction": recon_np
    }

def show_details(res, threshold):
    if res['is_defect']:
        st.error(f"🛑 ДЕФЕКТ (Score: {res['score']:.4f})")
    else:
        st.success(f"✅ OK (Score: {res['score']:.4f})")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.image(res['original'], caption="Оригінал", use_container_width=True)

    with c2:
        st.image(res['reconstruction'], caption="Реконструкція", clamp=True, use_container_width=True)

    with c3:
        fig, ax = plt.subplots()
        im = ax.imshow(
            res['heatmap'],
            cmap='jet',
            vmin=0,
            vmax=max(res['score'], threshold * 1.2)
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)

st.set_page_config("Nut QC System", layout="wide", page_icon="🔩")
st.title("🔩 Система контролю якості гайок")

model, device, saved_thresh = load_system()

if 'threshold' not in st.session_state:
    st.session_state.threshold = saved_thresh

if model is None:
    st.stop()

st.markdown(f"**Поточний поріг:** `{st.session_state.threshold:.4f}`")

with st.expander("⚙️ Налаштування"):
    unlock_settings = st.checkbox("🔓 Розблокувати зміну порогу чутливості")
    
    if unlock_settings:
        st.warning(f"⚠️ Увага: Зміна порогу вплине на визначення дефектів при наступному аналізі. Стандартне значення {saved_thresh:.4f}.")

    st.slider(
        "Threshold (Поріг чутливості)",
        0.005,
        0.9,
        step=0.001,
        key="threshold",
        format="%.4f",
        disabled=not unlock_settings 
    )


uploaded_files = st.file_uploader(
    "Завантажте зображення",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files and st.button("🚀 Запустити аналіз", type="primary"):
    results = []
    bar = st.progress(0)

    for i, file in enumerate(uploaded_files):
        file.seek(0) 
        img = Image.open(file).convert("RGB")
        res = analyze_image(img, model, device, st.session_state.threshold)

        results.append({
            "filename": file.name,
            "score": res['score'],
            "is_defect": res['is_defect'],
            "status": "⛔ БРАК" if res['is_defect'] else "✅ OK",
            "file_obj": file 
        })
        bar.progress((i + 1) / len(uploaded_files))
    
    st.session_state['analysis_results'] = pd.DataFrame(results)

if 'analysis_results' in st.session_state and uploaded_files:
    df = st.session_state['analysis_results']

    st.subheader("📊 Статистика")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Всього", len(df))
    c2.metric("OK", (~df.is_defect).sum())
    c3.metric("Брак", df.is_defect.sum(), delta_color="inverse")
    c4.metric("% браку", f"{df.is_defect.mean() * 100:.1f}%")

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.dataframe(df[['filename', 'status', 'score']], use_container_width=True)

    with col2:
        options = [
            f"{'🔴' if r.is_defect else '🟢'} {r.filename}"
            for r in df.itertuples()
        ]
        selected = st.selectbox("Інспектор", options)

        if selected:
            fname = selected.split(" ", 1)[1]
            
            row = df[df['filename'] == fname].iloc[0]
            file_obj = row['file_obj']

            file_obj.seek(0)
            img = Image.open(file_obj).convert("RGB")
            
            res = analyze_image(img, model, device, st.session_state.threshold)

            st.divider()
            show_details(res, st.session_state.threshold)

elif not uploaded_files and 'analysis_results' in st.session_state:
    del st.session_state['analysis_results']