from __future__ import annotations

import io
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image as RLImage, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ─────────────────────────────────────────
# SEMAPI BRAND COLORS & PALETTE
# ─────────────────────────────────────────
SEMAPI_BLUE   = colors.HexColor("#1B3A6B")
SEMAPI_ORANGE = colors.HexColor("#E8720C")
SEMAPI_LGRAY  = colors.HexColor("#F0F4FA")
SEMAPI_DGRAY  = colors.HexColor("#4A4A4A")
WHITE         = colors.white

PAL = {
    "bg": "#0F1117", "card": "#1C1F2E", "accent": "#4F8EF7",
    "danger": "#FF4B4B", "ok": "#00C48C", "text": "#E0E6F0", "muted": "#6B7280",
    "warning": "#F59E0B", "harmonic": "#FF6B6B",
}

# ─────────────────────────────────────────
# CATÁLOGO DE TIPOS DE MÁQUINAS SEMAPI
# ─────────────────────────────────────────
MACHINE_TYPES = {
    "— Seleccionar tipo de máquina —": [],
    "🔨 Molinos y equipos de trituración": [
        "Molino Zerma", "Molino Preabreaker", "Pulverizadores",
        "Granuladores", "Equipos tipo Cumberland", "Otro molino / triturador",
    ],
    "🌀 Extrusoras": [
        "Extrusora (línea general)", "Otro modelo de extrusora",
    ],
    "🔃 Mezcladores": [
        "Mezclador industrial", "Otro mezclador",
    ],
    "❄️ Enfriadores y torres de enfriamiento": [
        "Sistema enfriador", "Torre de enfriamiento", "Otro enfriador",
    ],
    "💨 Sopladores (Blowers)": [
        "Blower Pallman", "Blower Cumberland", "Blower HP3", "Soplador general",
    ],
    "🌡️ Secadores": [
        "Secador Yankee PM4", "Secador de arcilla", "Otro secador",
    ],
    "📦 Zarandas": [
        "Zaranda 620", "Otra zaranda",
    ],
    "🌬️ Ventiladores": [
        "Ventilador principal (Main Fan)", "Ventilador de filtro principal",
        "Ventilador de filtro de reparto", "Ventilador de premolienda",
        "Ventilador de lecho fluido", "Ventilador del quemador del secador",
        "Otro ventilador industrial",
    ],
    "🔧 Compresores": [
        "Compresor SABROE SMC", "Compresor MYCOM", "Compresor ABC",
        "Compresor AF B4000", "Compresor BELLIS", "Compresor Sullair",
        "Otro compresor",
    ],
    "⚡ Equipos de generación de energía": [
        "Generador CAT G3520H", "Alternador industrial", "Otro generador / alternador",
    ],
    "🚿 Bombas": [
        "Bomba mecánica", "Bomba de agua caliente", "Bomba de agua fría",
        "Bomba de condensado", "Bomba de envío a extrusora",
        "Bomba principal de prensa", "Bomba de recirculación", "Otra bomba",
    ],
    "⚙️ Sistemas de transmisión de potencia": [
        "Caja reductora (Gear Box)", "Motorreductor para molino principal",
        "Otro sistema de transmisión",
    ],
    "🌀 Manejadoras de aire": [
        "Manejadora de aire (climatización)", "Unidad de tratamiento de aire (UTA)",
        "Otra manejadora",
    ],
    "🔬 Separadores dinámicos": [
        "Separador dinámico industrial", "Otro separador",
    ],
}

# ─────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────
st.set_page_config(page_title="Monitor de Vibraciones", page_icon="⚙️",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(f"""
<style>
  html, body, [class*="css"] {{ background-color:{PAL['bg']}; color:{PAL['text']}; font-family:'Inter',sans-serif; }}
  .block-container{{padding-top:1.5rem}}
  .metric-card{{background:{PAL['card']};border-radius:12px;padding:20px 24px;border:1px solid #2A2D3E}}
  .metric-value{{font-size:2rem;font-weight:700;color:{PAL['accent']}}}
  .metric-label{{font-size:.82rem;color:{PAL['muted']};text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}}
  .status-ok{{color:{PAL['ok']};font-weight:600}}
  .status-alarm{{color:{PAL['danger']};font-weight:600}}
  .machine-badge{{
      background:linear-gradient(135deg,#1B3A6B,#2563EB);
      border-radius:8px;padding:10px 16px;margin:6px 0 14px 0;
      border-left:3px solid #E8720C;font-size:.88rem;font-weight:600;
      color:#E0E6F0;
  }}
  .harmonic-info{{
      background:{PAL['card']};border:1px solid #2A2D3E;border-left:3px solid #FF6B6B;
      border-radius:8px;padding:10px 14px;margin:4px 0;font-size:.82rem;color:{PAL['muted']};
  }}
  h1{{font-size:1.7rem!important;font-weight:700}}
  h2{{font-size:1.1rem!important;font-weight:600;color:{PAL['muted']}}}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# FILE FILTER
# ─────────────────────────────────────────
VALID_EXTENSIONS = ('.csv', '.txt', '.dat', '.asc', '.tsv', '.prn', '.out')

def is_valid_file(name: str) -> bool:
    base = name.split('/')[-1]
    if name.startswith('__MACOSX') or base.startswith('.') or name.endswith('/'):
        return False
    if '.' not in base:
        return True
    ext = base.rsplit('.', 1)[-1]
    if ext.isdigit():
        return True
    return base.endswith(VALID_EXTENSIONS)

# ─────────────────────────────────────────
# SIGNAL UTILS
# ─────────────────────────────────────────
def safe_kurtosis(x):
    x = x.astype(np.float64)
    xc = x - x.mean()
    var = np.mean(xc**2)
    return float(np.mean(xc**4) / var**2 - 3.) if var > 0 else np.nan

def compute_features(signal):
    s = signal.astype(np.float64)
    rms  = float(np.sqrt(np.mean(s**2)))
    peak = float(np.max(np.abs(s)))
    kurt = safe_kurtosis(s)
    cf   = peak / rms if rms > 0 else np.nan
    return {"rms": rms, "peak": peak, "kurtosis_excess": kurt, "crest_factor": cf}

def compute_fft(signal, fs_hz: float):
    """Devuelve (freqs_hz, magnitud_lineal) del espectro unilateral."""
    s = signal.astype(np.float64)
    n = len(s)
    if n < 4:
        return np.array([0.0]), np.array([0.0])
    window   = np.hanning(n)
    s_win    = s * window
    spectrum = np.abs(np.fft.rfft(s_win)) * 2.0 / n
    freqs    = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    return freqs, spectrum

def dominant_frequency(freqs, spectrum) -> float:
    if len(spectrum) < 2:
        return np.nan
    idx = int(np.argmax(spectrum[1:])) + 1
    return float(freqs[idx])

def compute_harmonics(rpm: float, n_harmonics: int, fs_hz: float) -> list[dict]:
    """
    Calcula las frecuencias armónicas a partir de las RPM.
    f_n = n * (RPM / 60)   [Hz]
    Devuelve una lista de dicts con orden, frecuencia y etiqueta.
    Solo incluye armónicos dentro del rango de Nyquist.
    """
    if rpm <= 0:
        return []
    f1 = rpm / 60.0          # frecuencia fundamental en Hz
    nyquist = fs_hz / 2.0
    harmonics = []
    for n in range(1, n_harmonics + 1):
        f_n = n * f1
        if f_n > nyquist:
            break
        label = f"{n}× = {f_n:.1f} Hz" if n > 1 else f"1× (Fund.) = {f_n:.1f} Hz"
        harmonics.append({"orden": n, "frecuencia_hz": f_n, "label": label})
    return harmonics

@st.cache_data(show_spinner=False)
def load_single_signal(zip_bytes: bytes, filename: str, sep_val: str, col_sig: str | None):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        with z.open(filename) as f:
            df_s, sig_col, _ = read_flexible(f, sep_val, col_sig)
    if col_sig and col_sig in df_s.columns:
        sig_col = col_sig
    elif sig_col not in df_s.columns:
        sig_col = df_s.columns[0]
    return df_s[sig_col].dropna().astype(np.float64).to_numpy()

def read_flexible(file_obj, sep_val: str, col_sig: str | None = None, usecols=None):
    raw = file_obj.read()
    try:
        df = pd.read_csv(io.BytesIO(raw), sep=sep_val, engine="python", dtype=np.float32, usecols=usecols)
        first_col = str(df.columns[0]).lstrip('-').replace('.', '').replace('e', '').replace('+','').replace('-','')
        if first_col.isdigit():
            raise ValueError("header numérico → sin encabezado")
        return df, col_sig, True
    except Exception:
        pass
    df = pd.read_csv(io.BytesIO(raw), sep=sep_val, engine="python", header=None, dtype=np.float32)
    df.columns = [f"Canal_{i+1}" for i in range(df.shape[1])]
    sig_col = col_sig if col_sig in df.columns else "Canal_1"
    return df, sig_col, False

@st.cache_data(show_spinner=False)
def process_zip_files(zip_bytes: bytes, sep_val: str, col_time: str, col_sig: str):
    rows = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        valid_files = sorted([n for n in z.namelist() if is_valid_file(n)])
        if not valid_files:
            return pd.DataFrame()

        bar   = st.progress(0, text="Iniciando procesamiento...")
        total = len(valid_files)

        for i, name in enumerate(valid_files):
            if i % 10 == 0:
                bar.progress((i + 1) / total, text=f"Procesando {i+1}/{total} — {name.split('/')[-1]}")

            with z.open(name) as f:
                try:
                    df_sig, sig_col, has_hdr = read_flexible(f, sep_val, col_sig)
                except Exception:
                    continue

            if sig_col not in df_sig.columns:
                sig_col = df_sig.columns[0]
            sig = df_sig[sig_col].dropna().astype(np.float64).to_numpy()
            if len(sig) == 0:
                continue

            if col_time == "Usar nombre del archivo":
                dt = name.split('/')[-1]
            elif col_time in df_sig.columns:
                try: dt = pd.to_datetime(df_sig[col_time].iloc[0])
                except Exception: dt = str(df_sig[col_time].iloc[0])
            else:
                dt = name.split('/')[-1]

            feats = compute_features(sig)
            freqs_f, spec_f = compute_fft(sig, fs_hz)
            dom_freq = dominant_frequency(freqs_f, spec_f)
            rows.append({"datetime": dt, "filename": name.split('/')[-1], "n_samples": sig.size,
                         "dominant_freq": dom_freq, **feats})

        bar.empty()
    return pd.DataFrame(rows)

def peek_zip(uploaded_zip, sep_val: str):
    try:
        with zipfile.ZipFile(uploaded_zip) as z:
            valid = [n for n in z.namelist() if is_valid_file(n)]
            if not valid: return None, []
            with z.open(valid[0]) as f:
                df_peek, _, _ = read_flexible(f, sep_val)
            return df_peek.head(5), df_peek.columns.tolist()
    except Exception:
        return None, []

# ─────────────────────────────────────────
# PLOTLY CHARTS
# ─────────────────────────────────────────
# Paleta de colores para las líneas armónicas (orden 1..N)
HARMONIC_COLORS = [
    "#FF6B6B",   # 1× fundamental — rojo
    "#FFB347",   # 2× — naranja
    "#FFFF66",   # 3× — amarillo
    "#B0F566",   # 4× — verde lima
    "#66F5D4",   # 5× — turquesa
    "#66B2FF",   # 6× — azul cielo
    "#CC99FF",   # 7× — violeta
    "#FF99CC",   # 8× — rosa
    "#FF6666",   # 9×
    "#FFA07A",   # 10×
]

def plotly_fft(freqs, spectrum, title="Espectro de Frecuencia", fs_hz=20_000, harmonics: list | None = None):
    """
    Gráfica interactiva del espectro FFT unilateral.
    harmonics: lista de dicts con claves 'frecuencia_hz' y 'label' (de compute_harmonics).
    """
    fig = go.Figure()

    # ── Espectro principal ──
    fig.add_trace(go.Scatter(
        x=freqs, y=spectrum,
        mode="lines", name="Amplitud",
        line=dict(color=PAL["accent"], width=1.4),
        fill="tozeroy", fillcolor="rgba(79,142,247,0.15)",
        hovertemplate="<b>%{x:.1f} Hz</b><br>Amplitud: %{y:.6f}<extra></extra>",
    ))

    # ── Frecuencia dominante ──
    if len(spectrum) > 1:
        dom_idx = int(np.argmax(spectrum[1:])) + 1
        fig.add_vline(
            x=float(freqs[dom_idx]),
            line_dash="dash", line_color=PAL["danger"],
            annotation_text=f"f_dom = {freqs[dom_idx]:.1f} Hz",
            annotation_font_color=PAL["danger"],
        )

    # ── Líneas armónicas ──
    if harmonics:
        for h in harmonics:
            n       = h["orden"]
            f_h     = h["frecuencia_hz"]
            lbl     = h["label"]
            color   = HARMONIC_COLORS[(n - 1) % len(HARMONIC_COLORS)]
            # Línea vertical
            fig.add_vline(
                x=f_h,
                line_dash="dot",
                line_color=color,
                line_width=1.6,
                annotation_text=lbl,
                annotation_font_color=color,
                annotation_font_size=9,
                annotation_position="top right" if n % 2 == 0 else "top left",
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=PAL["text"])),
        paper_bgcolor=PAL["card"], plot_bgcolor=PAL["card"],
        font=dict(color=PAL["text"], size=12),
        xaxis=dict(gridcolor="#2A2D3E", title="Frecuencia (Hz)", range=[0, fs_hz / 2]),
        yaxis=dict(gridcolor="#2A2D3E", title="Amplitud"),
        margin=dict(l=20, r=20, t=48, b=20), height=380,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig

def plotly_line(df, y_col, title, color, threshold=None, thr_label="Umbral"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["datetime"], y=df[y_col], mode="lines+markers", name=y_col,
                             line=dict(color=color, width=1.8), marker=dict(size=4),
                             hovertemplate="<b>%{x}</b><br>" + y_col + ": %{y:.4f}<extra></extra>"))
    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dash", line_color=PAL["danger"],
                      annotation_text=thr_label, annotation_font_color=PAL["danger"])
        mask = df[y_col] > threshold
        if mask.any():
            fig.add_trace(go.Scatter(x=df[mask]["datetime"], y=df[mask][y_col],
                                     mode="markers", marker=dict(color=PAL["danger"], size=6), name="⚠ Alarma"))
    is_cat = df["datetime"].dtype == "O"
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=PAL["text"])),
        paper_bgcolor=PAL["card"], plot_bgcolor=PAL["card"], font=dict(color=PAL["text"], size=12),
        xaxis=dict(gridcolor="#2A2D3E", title="Identificador / Tiempo", type="category" if is_cat else "-"),
        yaxis=dict(gridcolor="#2A2D3E"), legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=20, r=20, t=48, b=20), height=320,
    )
    return fig

def plotly_gauge_plumilla(valor_actual, umbral, titulo):
    max_val = max(valor_actual * 1.2, umbral * 1.5)
    if max_val == 0: max_val = 1

    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=valor_actual,
        title={'text': titulo, 'font': {'size': 16, 'color': PAL["text"]}},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': PAL["text"]},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': PAL["bg"],
            'borderwidth': 2,
            'bordercolor': "#2A2D3E",
            'steps': [
                {'range': [0, umbral * 0.8], 'color': PAL["ok"]},
                {'range': [umbral * 0.8, umbral], 'color': "#F59E0B"},
                {'range': [umbral, max_val], 'color': PAL["danger"]}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.75,
                'value': umbral
            }
        }
    ))

    prop  = min(max(valor_actual / max_val, 0), 1)
    theta = np.pi * (1 - prop)
    r     = 0.35
    x_tip = 0.5 + r * np.cos(theta)
    y_tip = 0.24 + r * np.sin(theta)
    path  = f"M 0.49 0.24 L {x_tip} {y_tip} L 0.51 0.24 Z"

    fig.update_layout(
        shapes=[
            dict(type="path", path=path, fillcolor="white", line_color="white", xref="paper", yref="paper"),
            dict(type="circle", x0=0.48, y0=0.22, x1=0.52, y1=0.26, fillcolor=PAL["card"], line_color="white", xref="paper", yref="paper"),
        ],
        annotations=[
            dict(x=0.5, y=0.10, xref="paper", yref="paper",
                 text=f"<b>{valor_actual:.4f}</b>", showarrow=False,
                 font=dict(size=22, color="white"))
        ],
        paper_bgcolor=PAL["card"],
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig

# ─────────────────────────────────────────
# PDF HELPERS
# ─────────────────────────────────────────
def make_fft_png(freqs, spectrum, fs_hz, harmonics=None):
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.plot(freqs, spectrum, color="#2563EB", linewidth=1.0)
    ax.fill_between(freqs, spectrum, alpha=0.2, color="#2563EB")
    if len(spectrum) > 1:
        dom_idx = int(np.argmax(spectrum[1:])) + 1
        ax.axvline(freqs[dom_idx], color="#CC3333", linestyle="--", linewidth=1,
                   label=f"f_dom = {freqs[dom_idx]:.1f} Hz")

    # Líneas armónicas en el PDF
    if harmonics:
        mpl_colors = ["#FF6B6B","#FF9900","#CCCC00","#33AA33","#00AACC",
                      "#3366FF","#9933FF","#FF66AA","#FF4444","#FF8866"]
        for h in harmonics:
            c = mpl_colors[(h["orden"] - 1) % len(mpl_colors)]
            ax.axvline(h["frecuencia_hz"], color=c, linestyle=":", linewidth=0.9,
                       label=h["label"])

    ax.legend(fontsize=6, loc="upper right")
    ax.set_xlim(0, fs_hz / 2)
    ax.set_xlabel("Frecuencia (Hz)", fontsize=8)
    ax.set_ylabel("Amplitud", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#F8F9FB")
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def make_chart_png(df, y_col, ylabel, color, threshold=None):
    fig, ax = plt.subplots(figsize=(7, 2.6))
    is_cat  = df["datetime"].dtype == "O"
    x_data  = np.arange(len(df)) if is_cat else df["datetime"].to_numpy()
    y_data  = df[y_col].to_numpy()

    ax.plot(x_data, y_data, color=color, linewidth=1.2, marker="o", markersize=2)
    if threshold is not None:
        ax.axhline(threshold, color="#CC3333", linestyle="--", linewidth=1, label=f"Umbral {threshold:.4f}")
        mask = y_data > threshold
        if mask.any():
            ax.scatter(x_data[mask], y_data[mask], color="#CC3333", s=12, zorder=5)
        ax.legend(fontsize=7)

    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel("Muestras / Tiempo", fontsize=8)
    if is_cat:
        step = max(1, len(df) // 10)
        ticks = np.arange(0, len(df), step)
        ax.set_xticks(ticks)
        ax.set_xticklabels(df["datetime"].iloc[ticks].astype(str), rotation=25, ha="right", fontsize=6)
    else:
        ax.tick_params(axis="x", rotation=25, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#F8F9FB")
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def semapi_styles():
    S = {}
    S["cover_title"] = ParagraphStyle("ct", fontSize=22, textColor=SEMAPI_BLUE, alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=6, leading=28)
    S["cover_sub"]   = ParagraphStyle("cs", fontSize=14, textColor=SEMAPI_BLUE, alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=4, leading=18)
    S["cover_body"]  = ParagraphStyle("cb", fontSize=11, textColor=SEMAPI_DGRAY, alignment=TA_CENTER, fontName="Helvetica", leading=16)
    S["sec_hdr"]     = ParagraphStyle("sh", fontSize=11, textColor=WHITE, fontName="Helvetica-Bold", alignment=TA_LEFT, spaceAfter=0, leading=14)
    S["body"]        = ParagraphStyle("bo", fontSize=9, textColor=SEMAPI_DGRAY, fontName="Helvetica", leading=13, alignment=TA_JUSTIFY)
    S["bold_blue"]   = ParagraphStyle("bb", fontSize=9, textColor=SEMAPI_BLUE, fontName="Helvetica-Bold", leading=13)
    return S

def _hf(canvas, doc, fecha_full):
    canvas.saveState()
    W, H = letter
    canvas.setFillColor(SEMAPI_BLUE); canvas.rect(0, H-50, W, 50, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 15); canvas.drawString(0.4*inch, H-32, "SEMAPI")
    canvas.setFont("Helvetica", 7);       canvas.drawString(0.4*inch, H-43, "Servicios de Mantenimiento e Ingeniería")
    canvas.setFont("Helvetica-Bold", 9);  canvas.drawRightString(W-0.4*inch, H-28, "INFORME DE ANÁLISIS DE VIBRACIONES")
    canvas.setFont("Helvetica", 8);       canvas.drawRightString(W-0.4*inch, H-40, fecha_full)
    canvas.setFillColor(SEMAPI_BLUE); canvas.rect(0, 0, W, 32, fill=1, stroke=0)
    canvas.setFillColor(WHITE); canvas.setFont("Helvetica", 7)
    canvas.drawCentredString(W/2, 18, "Dir.: Via 40 #82 - 47  |  Tels: 3788848 - 3003621 - 3731357  |  Barranquilla")
    canvas.drawCentredString(W/2, 8,  "E-mail: Gerenteservicios@semapicolombia.com / info@semapicolombia.com")
    canvas.drawRightString(W-0.3*inch, 18, f"Pág. {doc.page}")
    canvas.restoreState()

def section_box(label, S):
    tbl = Table([[Paragraph(label, S["sec_hdr"])]], colWidths=[6.9*inch])
    tbl.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), SEMAPI_BLUE), ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5), ("LEFTPADDING", (0,0), (-1,-1), 8)]))
    return tbl

# ─────────────────────────────────────────
# PDF GENERATOR
# ─────────────────────────────────────────
def generate_pdf_report(df, thr_rms, thr_kurt, sigma_mult, baseline_n,
                         alarm_rms, alarm_kurt, client_name, equipo, ingeniero,
                         fs_hz, machine_type_label, rpm_val,
                         fft_freqs=None, fft_spectrum=None, harmonics=None):
    buf       = io.BytesIO()
    fecha_full = datetime.now().strftime("%d/%m/%Y")
    fecha_str  = datetime.now().strftime("%B %Y").capitalize()
    S         = semapi_styles()

    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.6*inch, rightMargin=0.6*inch,
                            topMargin=0.85*inch, bottomMargin=0.65*inch)
    story = []

    # Portada
    story.append(Spacer(1, 1.2*inch))
    story.append(Paragraph("INFORME DE", S["cover_title"]))
    story.append(Paragraph("ANÁLISIS DE VIBRACIONES", S["cover_title"]))
    story.append(Spacer(1, 0.4*inch))
    story.append(HRFlowable(width="70%", thickness=2, color=SEMAPI_ORANGE, lineCap="round", spaceAfter=18, spaceBefore=4, hAlign="CENTER"))
    story.append(Paragraph(client_name.upper(), S["cover_sub"]))
    story.append(Spacer(1, 0.15*inch))
    if machine_type_label and machine_type_label != "— Seleccionar tipo de máquina —":
        story.append(Paragraph(f"Tipo de equipo: {machine_type_label}", S["cover_body"]))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(fecha_str, S["cover_body"]))
    story.append(Spacer(1, 1.0*inch))
    story.append(Paragraph("Análisis de degradación de maquinaria industrial", S["cover_body"]))
    story.append(PageBreak())

    # Introducción
    story.append(section_box("REF.: ANÁLISIS DE VIBRACIONES", S))
    story.append(Spacer(1, 8))
    rpm_info = f" Las RPM operativas configuradas son <b>{rpm_val:.0f} RPM</b> (frecuencia fundamental: <b>{rpm_val/60:.2f} Hz</b>)." if rpm_val > 0 else ""
    story.append(Paragraph(
        f"Se efectuó servicio de análisis de vibraciones al equipo <b>{equipo}</b>. "
        f"Se procesaron <b>{len(df)}</b> archivos de medición con una frecuencia de muestreo de "
        f"<b>{fs_hz/1000:.1f} kHz</b>, extrayendo indicadores estadísticos en dominio del tiempo.{rpm_info}", S["body"]))
    story.append(Spacer(1, 14))

    # Tabla de indicadores
    story.append(section_box("TABLA DE DATOS – INDICADORES ESTADÍSTICOS", S))
    story.append(Spacer(1, 6))
    hdr = ["Indicador", "Valor Inicial", "Valor Final", "Valor Máximo", "Umbral alarma"]
    rows_tbl = [
        ["RMS",                  f"{df['rms'].iloc[0]:.5f}",               f"{df['rms'].iloc[-1]:.5f}",              f"{df['rms'].max():.5f}",            f"{thr_rms:.5f}"],
        ["Kurtosis",             f"{df['kurtosis_excess'].iloc[0]:.3f}",  f"{df['kurtosis_excess'].iloc[-1]:.3f}", f"{df['kurtosis_excess'].max():.3f}", f"{thr_kurt:.3f}"],
        ["Crest Factor",         f"{df['crest_factor'].iloc[0]:.3f}",     f"{df['crest_factor'].iloc[-1]:.3f}",    f"{df['crest_factor'].max():.3f}",    "—"],
        ["Frec. Dominante (Hz)", f"{df['dominant_freq'].iloc[0]:.1f}",    f"{df['dominant_freq'].iloc[-1]:.1f}",   f"{df['dominant_freq'].max():.1f}",   "—"],
    ]
    ft = Table([hdr] + rows_tbl, colWidths=[1.9*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.1*inch], repeatRows=1)
    ft.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), SEMAPI_BLUE), ("TEXTCOLOR", (0,0), (-1,0), WHITE),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 8),
        ("ALIGN", (1,0), (-1,-1), "CENTER"), ("ALIGN", (0,0), (0,-1), "LEFT"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [SEMAPI_LGRAY, WHITE]),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING", (0,0), (0,-1), 6),
    ]))
    story.append(ft)
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<i>Baseline: primeros {baseline_n} archivos · Umbral = media + {sigma_mult:.1f}σ</i>", S["body"]))
    story.append(Spacer(1, 14))

    # Tabla de armónicos en el PDF (si aplica)
    if harmonics:
        story.append(section_box("ARMÓNICOS CALCULADOS A PARTIR DE RPM", S))
        story.append(Spacer(1, 6))
        harm_hdr  = ["Orden", "Frecuencia (Hz)", "Descripción"]
        harm_rows = [[str(h["orden"]), f"{h['frecuencia_hz']:.2f}", h["label"]] for h in harmonics]
        ht = Table([harm_hdr] + harm_rows, colWidths=[1.0*inch, 2.0*inch, 3.9*inch], repeatRows=1)
        ht.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), SEMAPI_BLUE), ("TEXTCOLOR", (0,0), (-1,0), WHITE),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 8),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [SEMAPI_LGRAY, WHITE]),
            ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#CCCCCC")),
            ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(ht)
        story.append(Spacer(1, 14))

    # Gráficas de tendencia
    for label, y_col, ylabel, color, thr in [
        ("GRÁFICA DE TENDENCIA – RMS",                "rms",             "RMS",           "#2563EB", thr_rms),
        ("GRÁFICA DE TENDENCIA – KURTOSIS (EXCESS)",   "kurtosis_excess", "Kurtosis",      "#7C3AED", thr_kurt),
        ("GRÁFICA DE TENDENCIA – CREST FACTOR",        "crest_factor",    "Crest Factor",  "#059669", None),
        ("GRÁFICA DE TENDENCIA – FRECUENCIA DOMINANTE","dominant_freq",   "Frec. Dom.(Hz)","#D97706", None),
    ]:
        story.append(section_box(label, S))
        story.append(Spacer(1, 6))
        story.append(RLImage(make_chart_png(df, y_col, ylabel, color, thr), width=6.5*inch, height=2.3*inch))
        story.append(Spacer(1, 14))

    # FFT con armónicos
    if fft_freqs is not None and fft_spectrum is not None:
        story.append(section_box("ESPECTRO DE FRECUENCIA – FFT (ARCHIVO DE REFERENCIA)", S))
        story.append(Spacer(1, 6))
        story.append(RLImage(make_fft_png(fft_freqs, fft_spectrum, fs_hz, harmonics),
                             width=6.5*inch, height=2.3*inch))
        story.append(Spacer(1, 14))

    # Diagnóstico
    n_rms  = len(alarm_rms)
    n_kurt = len(alarm_kurt)
    has_alarm = n_rms > 0 or n_kurt > 0

    if has_alarm:
        first_dt = str(alarm_rms.iloc[0]["datetime"]) if n_rms > 0 else str(alarm_kurt.iloc[0]["datetime"])
        diag = "Los niveles de vibración presentan una <b>tendencia creciente</b> que supera el umbral estadístico. "
        if n_rms  > 0: diag += f"RMS registra <b>{n_rms} eventos</b> sobre el umbral ({thr_rms:.5f}), primer evento: <b>{first_dt}</b>. "
        if n_kurt > 0: diag += f"Kurtosis registra <b>{n_kurt} eventos</b> anormales, indicativo de impactos mecánicos. "
        diag += "Se recomienda análisis espectral detallado (FFT/Envolvente)."
        reco = "Revisar la condición operativa del equipo. Programar inspección física y verificar lubricación."
    else:
        diag = "Los niveles de vibración se encuentran dentro de parámetros satisfactorios. No se observan indicios de degradación mecánica severa."
        reco = "Mantener esquema de monitoreo de condición periódico."

    story.append(KeepTogether([section_box("DIAGNÓSTICO", S), Spacer(1, 6), Paragraph(diag, S["body"])]))
    story.append(Spacer(1, 12))
    story.append(KeepTogether([section_box("RECOMENDACIÓN", S), Spacer(1, 6), Paragraph(reco, S["body"])]))
    story.append(Spacer(1, 24))

    story.append(HRFlowable(width="100%", thickness=0.5, color=SEMAPI_BLUE, spaceBefore=4, spaceAfter=10))
    story.append(Paragraph("Cualquier comentario o inquietud con gusto lo atenderemos.", S["body"]))
    story.append(Spacer(1, 18))
    fdata = [
        [Paragraph("<b>Realizó:</b>", S["bold_blue"]),     Paragraph("<b>Revisó:</b>", S["bold_blue"])],
        [Paragraph(ingeniero, S["body"]),                   Paragraph("Ing. XXXXXX", S["body"])],
        [Paragraph("Líder de Mantenimiento", S["body"]),   Paragraph("", S["body"])],
        [Paragraph("Especialista en Vibraciones", S["body"]), Paragraph("", S["body"])],
    ]
    ftbl = Table(fdata, colWidths=[3.4*inch, 3.4*inch])
    ftbl.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"), ("TOPPADDING", (0,0), (-1,-1), 3)]))
    story.append(ftbl)

    doc.build(story,
              onFirstPage=lambda c, d: _hf(c, d, fecha_full),
              onLaterPages=lambda c, d: _hf(c, d, fecha_full))
    return buf.getvalue()

@st.cache_data(show_spinner=False)
def get_cached_pdf(df, thr_rms, thr_kurt, sigma_mult, baseline_n,
                   alarm_rms, alarm_kurt, client_name, equipo, ingeniero,
                   fs_hz, machine_type_label, rpm_val,
                   fft_freqs=None, fft_spectrum=None, harmonics_tuple=None):
    harmonics = [{"orden": o, "frecuencia_hz": f, "label": l} for o, f, l in harmonics_tuple] if harmonics_tuple else None
    return generate_pdf_report(df, thr_rms, thr_kurt, sigma_mult, baseline_n,
                                alarm_rms, alarm_kurt, client_name, equipo, ingeniero,
                                fs_hz, machine_type_label, rpm_val,
                                fft_freqs, fft_spectrum, harmonics)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Ingesta de Datos (ZIP)")
    st.markdown("---")

    # ── SECCIÓN 0: TIPO DE MÁQUINA ───────────────────────────────────────────
    st.markdown("### 0. Tipo de Máquina")

    machine_category = st.selectbox(
        "Categoría de equipo",
        options=list(MACHINE_TYPES.keys()),
        index=0,
        help="Selecciona el tipo de máquina analizada. Afecta la portada del informe.",
    )

    machine_model = None
    if machine_category != "— Seleccionar tipo de máquina —":
        models = MACHINE_TYPES[machine_category]
        machine_model = st.selectbox("Modelo / subtipo", options=models)
        # Badge visual de confirmación
        cat_clean = machine_category.split(" ", 1)[-1]  # quita el emoji
        st.markdown(
            f'<div class="machine-badge">📌 {cat_clean}<br>'
            f'<span style="font-weight:400;font-size:.82rem;color:#A0AEC0">{machine_model}</span></div>',
            unsafe_allow_html=True,
        )

    machine_type_label = (
        f"{machine_category.split(' ',1)[-1]} – {machine_model}"
        if machine_model else "No especificado"
    )

    st.markdown("---")

    uploaded_zip = st.file_uploader(
        "📦 Sube un archivo .zip con tus CSV/TXT/DAT o archivos IMS",
        type=["zip"],
    )

    cols     = []
    sep_val  = ","
    col_time = "Usar nombre del archivo"
    col_sig  = None
    df_peek  = None

    if uploaded_zip:
        st.markdown("### 1. Formato de Archivo")
        sep_dict = {
            "Tabulador (\\t)": "\t",
            "Coma (,)":        ",",
            "Punto y coma (;)": ";",
            "Espacio ( )":     r"\s+",
        }

        detected_name = "Tabulador (\\t)"
        try:
            with zipfile.ZipFile(uploaded_zip) as z:
                valid = [n for n in z.namelist() if is_valid_file(n)]
                if valid:
                    with z.open(valid[0]) as f:
                        sample = f.read(2000).decode('utf-8', errors='ignore')
                    counts = {"\t": sample.count('\t'), ",": sample.count(','),
                              ";": sample.count(';'), r"\s+": sample.count(' ')}
                    detected = max(counts, key=counts.get)
                    if counts[detected] > 0:
                        map_name = {"\t": "Tabulador (\\t)", ",": "Coma (,)",
                                    ";": "Punto y coma (;)", r"\s+": "Espacio ( )"}
                        detected_name = map_name[detected]
                        st.caption(f"🔍 Separador detectado: **{detected_name}**")
        except Exception:
            pass

        sep_choice = st.selectbox("Separador de columnas", list(sep_dict.keys()),
                                  index=list(sep_dict.keys()).index(detected_name))
        sep_val = sep_dict[sep_choice]

        df_peek, cols = peek_zip(uploaded_zip, sep_val)
        if df_peek is None:
            st.error("No se encontraron archivos válidos en el ZIP.")
        elif not cols:
            st.warning("No se pudieron detectar las columnas. Verifica el separador.")

    if cols:
        st.markdown("### 2. Mapeo de Columnas")
        col_sig  = st.selectbox("🎯 Columna de Señal (Aceleración)", cols)
        col_time = st.selectbox(
            "⏱️ Eje X (Tendencia)", ["Usar nombre del archivo"] + cols,
            help="Si cada archivo es una medición, usa el nombre del archivo para armar la tendencia.",
        )

    st.markdown("---")
    st.markdown("### 3. Parámetros Mecánicos")

    fs_hz = st.number_input("Frecuencia de Muestreo (Hz)",
                             min_value=100, max_value=100_000, value=20_000, step=100)

    # ── RPM y Armónicos ──────────────────────────────────────────────────────
    rpm_val = st.number_input(
        "RPM de operación",
        min_value=0, max_value=100_000, value=0, step=10,
        help="Velocidad de giro del eje. Con valor > 0 se calcularán y mostrarán los armónicos en la FFT.",
    )

    n_harmonics = 1  # valor por defecto
    if rpm_val > 0:
        n_harmonics = st.slider(
            "Número de armónicos a mostrar",
            min_value=1, max_value=10, value=5,
            help="Cantidad de múltiplos de la frecuencia fundamental (1×, 2×, 3×… n×) a marcar en el espectro.",
        )
        f1_hz = rpm_val / 60.0
        st.markdown(
            f'<div class="harmonic-info">'
            f'⚙️ Frecuencia fundamental: <b>{f1_hz:.2f} Hz</b><br>'
            f'Armónicos: 1× a {n_harmonics}×</div>',
            unsafe_allow_html=True,
        )

    baseline_pct = st.slider("Baseline (% inicial de datos)", 5, 40, 20, 5)
    sigma_mult   = st.slider("Multiplicador σ (Alarmas)", 1.0, 5.0, 3.0, 0.5)

    st.markdown("### 4. Datos del Informe")
    client_name  = st.text_input("Cliente",  value="PLANTA INDUSTRIAL")
    equipo_name  = st.text_input("Equipo",   value="Bomba Centrífuga - Lado Acople")
    ingeniero_nm = st.text_input("Realizó",  value="Ing. ")

    if st.button("▶  Procesar y Analizar", use_container_width=True, type="primary"):
        st.session_state.datos_procesados = True

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("# ⚙️ Dashboard de Monitoreo de Condición")
st.markdown("## Análisis de Tendencias de Vibración · SEMAPI")
st.markdown("---")

# Mostrar badge de máquina seleccionada en la pantalla principal
if machine_category != "— Seleccionar tipo de máquina —":
    col_badge, _ = st.columns([3, 5])
    with col_badge:
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#1B3A6B22,#2563EB22);'
            f'border:1px solid #2563EB55;border-left:4px solid #E8720C;border-radius:8px;'
            f'padding:10px 16px;margin-bottom:12px;">'
            f'<span style="font-size:.78rem;color:#6B7280;text-transform:uppercase;letter-spacing:.05em;">Equipo analizado</span><br>'
            f'<span style="font-weight:700;font-size:1rem;color:#E0E6F0">{machine_category.split(" ",1)[-1]}</span><br>'
            f'<span style="font-size:.88rem;color:#A0AEC0">{machine_model}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

if df_peek is not None and col_sig:
    with st.expander("🔍 Vista previa del primer archivo del ZIP"):
        st.dataframe(df_peek, use_container_width=True)

if not st.session_state.get("datos_procesados", False):
    st.info("👈 Sube un archivo .zip, selecciona el tipo de máquina, mapea tus columnas y presiona **Procesar y Analizar**.")
    st.stop()

# ─────────────────────────────────────────
# PROCESAMIENTO
# ─────────────────────────────────────────
with st.spinner("Descomprimiendo y extrayendo indicadores..."):
    try:
        df = process_zip_files(uploaded_zip.getvalue(), sep_val, col_time, col_sig)
    except Exception as e:
        st.error(f"Error al procesar los archivos: {e}")
        st.stop()

if df.empty:
    st.warning("No se extrajeron datos. Verifica la columna de señal y el separador.")
    st.stop()

# Thresholds automáticos
bn = max(2, int(len(df) * baseline_pct / 100))
st.caption(f"📐 Baseline automático: primeros **{bn} archivos** ({baseline_pct}% de {len(df)})")

base_rms        = df["rms"].iloc[:bn]
mu_r, sd_r      = base_rms.mean(), base_rms.std(ddof=1) if bn > 1 else 0.
base_kurt       = df["kurtosis_excess"].iloc[:bn]
mu_k, sd_k      = base_kurt.mean(), base_kurt.std(ddof=1) if bn > 1 else 0.
thr_rms         = mu_r + sigma_mult * sd_r
thr_kurt        = mu_k + sigma_mult * sd_k
alarm_rms       = df[df["rms"]             > thr_rms]
alarm_kurt      = df[df["kurtosis_excess"] > thr_kurt]

# ─────────────────────────────────────────
# KPI CARDS Y TACÓMETROS
# ─────────────────────────────────────────
st.markdown("### 📊 Indicadores Globales")

k1, k2, k3, k4 = st.columns(4)

def kpi(col, label, value, fmt=".4f"):
    col.markdown(f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value:{fmt}}</div>
    </div>""", unsafe_allow_html=True)

kpi(k1, "Archivos (Eventos)", len(df), "d")
kpi(k2, "RMS Máx", df["rms"].max())
kpi(k3, "Kurtosis Máx", df["kurtosis_excess"].max(), ".2f")
kpi(k4, f"Umbral RMS ({sigma_mult}σ)", thr_rms)

st.markdown("<br>", unsafe_allow_html=True)

t1, t2 = st.columns(2)
with t1:
    st.plotly_chart(plotly_gauge_plumilla(df["rms"].max(), thr_rms, "Estado Severidad RMS"), use_container_width=True)
with t2:
    st.plotly_chart(plotly_gauge_plumilla(df["kurtosis_excess"].max(), thr_kurt, "Estado Impactos (Kurtosis)"), use_container_width=True)

# ─────────────────────────────────────────
# CHARTS DE TENDENCIA
# ─────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(plotly_line(df, "rms", "Tendencia RMS (Energía Global)",
                                PAL["accent"], thr_rms, f"Umbral {sigma_mult:.1f}σ"), use_container_width=True)
with c2:
    st.plotly_chart(plotly_line(df, "kurtosis_excess", "Tendencia Kurtosis (Impulsividad)",
                                "#A78BFA", thr_kurt, f"Umbral {sigma_mult:.1f}σ"), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(plotly_line(df, "crest_factor", "Tendencia Factor de Cresta (Picos)", "#34D399"), use_container_width=True)
with c4:
    st.plotly_chart(plotly_line(df, "dominant_freq", "Tendencia Frecuencia Dominante (Hz)", "#F59E0B"), use_container_width=True)

# ─────────────────────────────────────────
# FFT SPECTRAL ANALYSIS — con armónicos
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("### 📡 Análisis Espectral – FFT")

# Panel de armónicos calculados
harmonics_list = []
if rpm_val > 0:
    harmonics_list = compute_harmonics(rpm_val, n_harmonics, fs_hz)
    if harmonics_list:
        with st.expander(f"🎯 Armónicos calculados ({len(harmonics_list)} líneas) — {rpm_val:.0f} RPM", expanded=True):
            cols_harm = st.columns(min(len(harmonics_list), 5))
            for idx, h in enumerate(harmonics_list):
                color = HARMONIC_COLORS[(h["orden"] - 1) % len(HARMONIC_COLORS)]
                cols_harm[idx % 5].markdown(
                    f'<div style="background:{PAL["card"]};border:1px solid #2A2D3E;'
                    f'border-top:3px solid {color};border-radius:6px;padding:8px 10px;'
                    f'text-align:center;margin:2px;">'
                    f'<div style="font-size:.72rem;color:#6B7280">{h["orden"]}× armónico</div>'
                    f'<div style="font-size:1.1rem;font-weight:700;color:{color}">{h["frecuencia_hz"]:.1f} Hz</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.warning("⚠️ Todos los armónicos quedan fuera del rango de Nyquist con la fs configurada.")
else:
    st.info("💡 Ingresa las **RPM de operación** en la barra lateral para superponer los armónicos sobre el espectro.", icon="ℹ️")

fft_file = st.selectbox(
    "Selecciona un archivo para ver su espectro de frecuencia",
    df["filename"].tolist(), index=0, key="fft_selector",
)

fft_freqs_viz = fft_spectrum_viz = None

if fft_file:
    with zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue())) as z_names:
        all_valid = [n for n in z_names.namelist() if is_valid_file(n)]
    full_name = next((n for n in all_valid if n.split("/")[-1] == fft_file), fft_file)

    with st.spinner("Calculando FFT…"):
        sig_fft = load_single_signal(uploaded_zip.getvalue(), full_name, sep_val, col_sig)

    if len(sig_fft) >= 4:
        fft_freqs_viz, fft_spectrum_viz = compute_fft(sig_fft, fs_hz)
        dom_f = dominant_frequency(fft_freqs_viz, fft_spectrum_viz)

        fi1, fi2, fi3 = st.columns(3)
        fi1.metric("Muestras", f"{len(sig_fft):,}")
        fi2.metric("Resolución espectral", f"{fs_hz / len(sig_fft):.2f} Hz")
        fi3.metric("Frecuencia dominante", f"{dom_f:.1f} Hz")

        st.plotly_chart(
            plotly_fft(
                fft_freqs_viz, fft_spectrum_viz,
                title=f"Espectro FFT — {fft_file}",
                fs_hz=fs_hz,
                harmonics=harmonics_list if harmonics_list else None,
            ),
            use_container_width=True,
        )

        # Detección automática de coincidencias armónico ↔ pico espectral
        if harmonics_list and len(fft_spectrum_viz) > 1:
            tol_hz = (fs_hz / len(sig_fft)) * 3   # tolerancia = 3 bins espectrales
            matches = []
            for h in harmonics_list:
                mask_near = np.abs(fft_freqs_viz - h["frecuencia_hz"]) <= tol_hz
                if mask_near.any():
                    local_peak = float(fft_spectrum_viz[mask_near].max())
                    global_max = float(fft_spectrum_viz[1:].max())
                    pct = local_peak / global_max * 100 if global_max > 0 else 0.0
                    if pct > 5.0:   # solo reportar si el pico es notable (> 5% del máximo global)
                        matches.append(f"**{h['label']}** → pico local = {pct:.1f}% del máximo espectral")
            if matches:
                st.warning("⚠️ **Coincidencias armónico–espectro detectadas:**\n\n" + "\n\n".join(f"- {m}" for m in matches))
    else:
        st.warning("La señal seleccionada tiene muy pocas muestras para calcular la FFT.")

# ─────────────────────────────────────────
# ALARM PANEL
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("### 🚨 Panel de Alarmas y Excepciones")
ca, cb = st.columns(2)

with ca:
    st.markdown(f"**RMS — {len(alarm_rms)} eventos** (umbral = `{thr_rms:.5f}`)")
    if alarm_rms.empty:
        st.markdown("<span class='status-ok'>✔ Sin alarmas RMS</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='status-alarm'>Primera alarma: {alarm_rms.iloc[0]['datetime']}</span>", unsafe_allow_html=True)
        with st.expander("Ver detalle"):
            st.dataframe(alarm_rms[["datetime", "filename", "rms"]], use_container_width=True, hide_index=True)

with cb:
    st.markdown(f"**Kurtosis — {len(alarm_kurt)} eventos** (umbral = `{thr_kurt:.4f}`)")
    if alarm_kurt.empty:
        st.markdown("<span class='status-ok'>✔ Sin alarmas Kurtosis</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='status-alarm'>Primera alarma: {alarm_kurt.iloc[0]['datetime']}</span>", unsafe_allow_html=True)
        with st.expander("Ver detalle"):
            st.dataframe(alarm_kurt[["datetime", "filename", "kurtosis_excess"]], use_container_width=True, hide_index=True)

# ─────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────
st.markdown("---")
e1, e2 = st.columns(2)

with e1:
    st.download_button(
        "⬇️  Descargar Histórico CSV",
        data=df.to_csv(index=False).encode(),
        file_name="tendencia_vibraciones.csv",
        mime="text/csv",
        use_container_width=True,
    )

with e2:
    with st.spinner("Compilando reporte técnico PDF..."):
        # Convertimos la lista de dicts a tupla de tuplas para que sea hasheable por st.cache_data
        harmonics_tuple = tuple(
            (h["orden"], h["frecuencia_hz"], h["label"]) for h in harmonics_list
        ) if harmonics_list else None
        # Reconstruimos como list of dicts dentro de get_cached_pdf
        pdf_bytes = get_cached_pdf(
            df, thr_rms, thr_kurt, sigma_mult, bn,
            alarm_rms, alarm_kurt, client_name, equipo_name,
            ingeniero_nm, fs_hz, machine_type_label, rpm_val,
            fft_freqs_viz, fft_spectrum_viz, harmonics_tuple,
        )
    st.download_button(
        "📄  Descargar Informe Técnico PDF",
        data=pdf_bytes,
        file_name=f"Reporte_Vibraciones_{equipo_name.replace(' ', '_')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
