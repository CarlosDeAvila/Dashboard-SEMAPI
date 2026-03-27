from __future__ import annotations

import io
import zipfile
from pathlib import Path
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
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
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
  h1{{font-size:1.7rem!important;font-weight:700}}
  h2{{font-size:1.1rem!important;font-weight:600;color:{PAL['muted']}}}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIGNAL UTILS
# ─────────────────────────────────────────
def safe_kurtosis(x):
    x = x.astype(np.float64); xc = x - x.mean(); var = np.mean(xc**2)
    return float(np.mean(xc**4)/var**2 - 3.) if var > 0 else np.nan

def compute_features(signal):
    s = signal.astype(np.float64)
    rms = float(np.sqrt(np.mean(s**2))); peak = float(np.max(np.abs(s)))
    kurt = safe_kurtosis(s); cf = peak/rms if rms > 0 else np.nan
    return {"rms": rms, "peak": peak, "kurtosis_excess": kurt, "crest_factor": cf}

@st.cache_data(show_spinner=False)
def process_zip_files(zip_bytes, sep_val, col_time, col_sig):
    rows = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        valid_files = sorted([
            n for n in z.namelist()
            if n.endswith(('.csv', '.txt')) and not n.startswith('__MACOSX')
        ])

        if not valid_files:
            return pd.DataFrame()

        bar = st.progress(0, text="Iniciando procesamiento...")
        total = len(valid_files)

        for i, name in enumerate(valid_files):
            # Actualiza la barra cada 10 archivos
            if i % 10 == 0:
                bar.progress((i + 1) / total, 
                             text=f"Procesando archivo {i+1} de {total} — {name.split('/')[-1]}")

            with z.open(name) as f:
                usecols = [col_sig] if col_time == "Usar nombre del archivo" else [col_time, col_sig]
                try:
                    # float32 usa la mitad de RAM que float64
                    df_sig = pd.read_csv(f, sep=sep_val, usecols=usecols,
                                         engine="python", dtype=np.float32)
                except Exception:
                    continue

            sig = df_sig[col_sig].dropna().astype(np.float64).to_numpy()
            if len(sig) == 0:
                continue

            if col_time == "Usar nombre del archivo":
                dt = name.split('/')[-1]
            else:
                try:
                    dt = pd.to_datetime(df_sig[col_time].iloc[0])
                except Exception:
                    dt = str(df_sig[col_time].iloc[0])

            feats = compute_features(sig)
            rows.append({
                "datetime": dt,
                "filename": name.split('/')[-1],
                "n_samples": sig.size,
                **feats
            })

        bar.empty()  # Limpia la barra al terminar

    return pd.DataFrame(rows)

# ─────────────────────────────────────────
# PLOTLY CHARTS
# ─────────────────────────────────────────
def plotly_line(df, y_col, title, color, threshold=None, thr_label="Umbral"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["datetime"], y=df[y_col], mode="lines+markers", name=y_col,
                             line=dict(color=color, width=1.8), marker=dict(size=4),
                             hovertemplate="<b>%{x}</b><br>"+y_col+": %{y:.4f}<extra></extra>"))
    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dash", line_color=PAL["danger"],
                      annotation_text=thr_label, annotation_font_color=PAL["danger"])
        mask = df[y_col] > threshold
        if mask.any():
            fig.add_trace(go.Scatter(x=df[mask]["datetime"], y=df[mask][y_col],
                                     mode="markers", marker=dict(color=PAL["danger"],size=6),
                                     name="⚠ Alarma"))
    # Check if x-axis is categorical (strings)
    is_categorical = df['datetime'].dtype == 'O'
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=PAL["text"])),
        paper_bgcolor=PAL["card"], plot_bgcolor=PAL["card"], font=dict(color=PAL["text"], size=12),
        xaxis=dict(gridcolor="#2A2D3E", title="Identificador / Tiempo", type='category' if is_categorical else '-'),
        yaxis=dict(gridcolor="#2A2D3E"), legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=20,r=20,t=48,b=20), height=320,
    )
    return fig

# ─────────────────────────────────────────
# PDF HELPERS
# ─────────────────────────────────────────
def make_chart_png(df, y_col, ylabel, color, threshold=None):
    fig, ax = plt.subplots(figsize=(7, 2.6))
    
    is_categorical = df['datetime'].dtype == 'O'
    x_data = range(len(df)) if is_categorical else df["datetime"]
    
    ax.plot(x_data, df[y_col], color=color, linewidth=1.2, marker='o', markersize=2)
    
    if threshold is not None:
        ax.axhline(threshold, color="#CC3333", linestyle="--", linewidth=1, label=f"Umbral {threshold:.4f}")
        mask = df[y_col] > threshold
        if mask.any():
            ax.scatter(np.array(x_data)[mask], df.loc[mask,y_col], color="#CC3333", s=12, zorder=5)
        ax.legend(fontsize=7)
        
    ax.set_ylabel(ylabel, fontsize=8); ax.set_xlabel("Muestras / Tiempo", fontsize=8)
    if is_categorical:
        ax.set_xticks(range(0, len(df), max(1, len(df)//10))) # Muestra solo algunos labels si son muchos archivos
        ax.set_xticklabels([df["datetime"].iloc[i] for i in ax.get_xticks()], rotation=25, ha='right', fontsize=6)
    else:
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, alpha=0.3); ax.set_facecolor("#F8F9FB")
    fig.patch.set_facecolor("white"); fig.tight_layout(pad=0.5)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
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
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),SEMAPI_BLUE), ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5), ("LEFTPADDING",(0,0),(-1,-1),8)]))
    return tbl

# ─────────────────────────────────────────
# MAIN PDF GENERATOR
# ─────────────────────────────────────────
def generate_pdf_report(df, thr_rms, thr_kurt, sigma_mult, baseline_n, alarm_rms, alarm_kurt, client_name, equipo, ingeniero, fs_hz):
    buf = io.BytesIO()
    fecha_full = datetime.now().strftime("%d/%m/%Y")
    fecha_str  = datetime.now().strftime("%B %Y").capitalize()
    S = semapi_styles()

    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=0.6*inch, rightMargin=0.6*inch, topMargin=0.85*inch, bottomMargin=0.65*inch)
    story = []

    story.append(Spacer(1, 1.2*inch))
    story.append(Paragraph("INFORME DE", S["cover_title"]))
    story.append(Paragraph("ANÁLISIS DE VIBRACIONES", S["cover_title"]))
    story.append(Spacer(1, 0.4*inch))
    story.append(HRFlowable(width="70%", thickness=2, color=SEMAPI_ORANGE, lineCap="round", spaceAfter=18, spaceBefore=4, hAlign="CENTER"))
    story.append(Paragraph(client_name.upper(), S["cover_sub"]))
    story.append(Spacer(1, 0.25*inch))
    story.append(Paragraph(fecha_str, S["cover_body"]))
    story.append(Spacer(1, 1.2*inch))
    story.append(Paragraph("Análisis de degradación de maquinaria industrial", S["cover_body"]))
    story.append(PageBreak())

    story.append(section_box("REF.: ANÁLISIS DE VIBRACIONES", S))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Se efectuó servicio de análisis de vibraciones al equipo <b>{equipo}</b>. "
        f"Se procesaron <b>{len(df)}</b> archivos de medición temporal con una frecuencia de muestreo configurada de "
        f"<b>{fs_hz/1000:.1f} kHz</b>, extrayendo indicadores estadísticos en dominio del tiempo para evaluar el estado mecánico.",
        S["body"]
    ))
    story.append(Spacer(1, 14))

    story.append(section_box("TABLA DE DATOS – INDICADORES ESTADÍSTICOS", S))
    story.append(Spacer(1, 6))
    hdr = ["Indicador", "Valor Inicial", "Valor Final", "Valor Máximo", "Umbral alarma"]
    rows = [
        ["RMS (aceleración)",   f"{df['rms'].iloc[0]:.5f}",             f"{df['rms'].iloc[-1]:.5f}",             f"{df['rms'].max():.5f}",             f"{thr_rms:.5f}"],
        ["Kurtosis (excess)",   f"{df['kurtosis_excess'].iloc[0]:.3f}", f"{df['kurtosis_excess'].iloc[-1]:.3f}", f"{df['kurtosis_excess'].max():.3f}", f"{thr_kurt:.3f}"],
        ["Crest Factor",        f"{df['crest_factor'].iloc[0]:.3f}",   f"{df['crest_factor'].iloc[-1]:.3f}",   f"{df['crest_factor'].max():.3f}",   "—"],
    ]
    ft = Table([hdr]+rows, colWidths=[1.9*inch,1.0*inch,1.0*inch,1.0*inch,1.1*inch], repeatRows=1)
    ft.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),SEMAPI_BLUE),("TEXTCOLOR",(0,0),(-1,0),WHITE),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),("ALIGN",(0,0),(0,-1),"LEFT"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[SEMAPI_LGRAY,WHITE]),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CCCCCC")),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4), ("LEFTPADDING",(0,0),(0,-1),6),
    ]))
    story.append(ft)
    story.append(Spacer(1,4))
    story.append(Paragraph(f"<i>Baseline: primeras {baseline_n} muestras · Umbral = media + {sigma_mult:.1f}σ</i>", S["body"]))
    story.append(Spacer(1,14))

    for label, y_col, ylabel, color, thr in [
        ("GRÁFICA DE TENDENCIA – RMS",              "rms",              "RMS",       "#2563EB", thr_rms),
        ("GRÁFICA DE TENDENCIA – KURTOSIS (EXCESS)","kurtosis_excess",  "Kurtosis",  "#7C3AED", thr_kurt),
        ("GRÁFICA DE TENDENCIA – CREST FACTOR",     "crest_factor",     "Crest Factor","#059669", None),
    ]:
        story.append(section_box(label, S))
        story.append(Spacer(1,6))
        img_buf = make_chart_png(df, y_col, ylabel, color, thr)
        story.append(RLImage(img_buf, width=6.5*inch, height=2.3*inch))
        story.append(Spacer(1,14))

    n_rms  = len(alarm_rms)
    n_kurt = len(alarm_kurt)
    has_alarm = n_rms > 0 or n_kurt > 0

    if has_alarm:
        candidates = []
        if n_rms  > 0: candidates.append(str(alarm_rms.iloc[0]["datetime"]))
        if n_kurt > 0: candidates.append(str(alarm_kurt.iloc[0]["datetime"]))
        first_dt = candidates[0] # Simplificado para manejar strings o datetimes
        diag = (f"Los niveles de vibración presentan una <b>tendencia creciente</b> que supera el umbral estadístico. ")
        if n_rms  > 0: diag += f"El indicador RMS registra <b>{n_rms} eventos</b> sobre el umbral ({thr_rms:.5f}), primer evento en: <b>{first_dt}</b>. "
        if n_kurt > 0: diag += f"La Kurtosis registra <b>{n_kurt} eventos</b> anormales, indicativo de posibles impactos mecánicos. "
        diag += "Se recomienda análisis espectral detallado (FFT/Envolvente)."
        reco = "Revisar la condición operativa del equipo. Programar inspección física y verificar lubricación."
    else:
        diag = (f"Los niveles de vibración se encuentran dentro de parámetros satisfactorios según el baseline. No se observan indicios de degradación mecánica severa en la tendencia.")
        reco = "Mantener esquema de monitoreo de condición periódico."

    story.append(KeepTogether([section_box("DIAGNÓSTICO", S), Spacer(1,6), Paragraph(diag, S["body"])]))
    story.append(Spacer(1,12))
    story.append(KeepTogether([section_box("RECOMENDACIÓN", S), Spacer(1,6), Paragraph(reco, S["body"])]))
    story.append(Spacer(1,24))

    story.append(HRFlowable(width="100%", thickness=0.5, color=SEMAPI_BLUE, spaceBefore=4, spaceAfter=10))
    story.append(Paragraph("Cualquier comentario o inquietud con gusto lo atenderemos.", S["body"]))
    story.append(Spacer(1,18))
    fdata = [
        [Paragraph("<b>Realizó:</b>", S["bold_blue"]),  Paragraph("<b>Revisó:</b>", S["bold_blue"])],
        [Paragraph(ingeniero, S["body"]),                Paragraph("Ing. Eloy Balza", S["body"])],
        [Paragraph("Líder de Mantenimiento", S["body"]),Paragraph("", S["body"])],
        [Paragraph("Especialista en Vibraciones", S["body"]), Paragraph("", S["body"])],
    ]
    ftbl = Table(fdata, colWidths=[3.4*inch, 3.4*inch])
    ftbl.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),3)]))
    story.append(ftbl)

    doc.build(story, onFirstPage=lambda c,d: _hf(c,d,fecha_full), onLaterPages=lambda c,d: _hf(c,d,fecha_full))
    return buf.getvalue()

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Ingesta de Datos (ZIP)")
    st.markdown("---")
    
    uploaded_zip = st.file_uploader("📦 Sube un archivo .zip con tus CSV/TXT", type=["zip"])
    
    # Valores por defecto para prevenir errores si no hay ZIP
    cols = []
    sep_val = ","
    col_time = "Usar nombre del archivo"
    col_sig = None
    
    if uploaded_zip:
        st.markdown("### 1. Formato de Archivo")
        sep_dict = {"Coma (,)": ",", "Punto y coma (;)": ";", "Tabulador (\\t)": "\t", "Espacio ( )": r"\s+"}
        sep_choice = st.selectbox("Separador de columnas", list(sep_dict.keys()))
        sep_val = sep_dict[sep_choice]
        
        # PEEK: Leer sigilosamente las columnas del primer archivo
        try:
            with zipfile.ZipFile(uploaded_zip) as z:
                valid_files = [n for n in z.namelist() if n.endswith(('.csv', '.txt')) and not n.startswith('__MACOSX')]
                if valid_files:
                    with z.open(valid_files[0]) as f:
                        df_peek = pd.read_csv(f, sep=sep_val, engine="python", nrows=2)
                        cols = df_peek.columns.tolist()
                else:
                    st.error("El ZIP no contiene archivos .csv o .txt válidos.")
        except Exception as e:
            st.warning("No se pudieron detectar las columnas. Verifica el separador.")
    
    if cols:
        st.markdown("### 2. Mapeo de Columnas")
        col_sig = st.selectbox("🎯 Columna de Señal (Aceleración)", cols)
        col_time = st.selectbox("⏱️ Eje X (Tendencia)", ["Usar nombre del archivo"] + cols, 
                                help="Si tu archivo no tiene una fecha real por cada muestra, usa el nombre del archivo para armar la tendencia.")
        # col_rpm = st.selectbox("🔄 Columna de RPM (Opcional)", ["-- Ninguna --"] + cols) # Preparado para el futuro
    
    st.markdown("---")
    st.markdown("### 3. Parámetros Mecánicos")
    fs_hz = st.number_input("Frecuencia de Muestreo (Hz)", min_value=100, max_value=100000, value=25600, step=100)
    baseline_pct = st.slider("Baseline (% inicial de datos)", 5, 40, 20, 5,
                          help="Porcentaje de los primeros archivos usados como referencia de condición normal.")
    sigma_mult = st.slider("Multiplicador σ (Alarmas)", 1.0, 5.0, 3.0, 0.5)
    
    st.markdown("### 4. Datos del Informe")
    client_name  = st.text_input("Cliente",  value="PLANTA INDUSTRIAL")
    equipo_name  = st.text_input("Equipo",   value="Bomba Centrífuga - Lado Acople")
    ingeniero_nm = st.text_input("Realizó",  value="Ing. ")
    
    analyze_btn = st.button("▶  Procesar y Analizar", use_container_width=True, type="primary")

# ─────────────────────────────────────────
# HEADER & MAIN EXECUTION
# ─────────────────────────────────────────
st.markdown("# ⚙️ Dashboard de Monitoreo de Condición")
st.markdown("## Análisis de Tendencias de Vibración · SEMAPI")
st.markdown("---")

if cols and col_sig:
    with st.expander("🔍 Vista previa del primer archivo del ZIP"):
        st.dataframe(df_peek, use_container_width=True)

if not analyze_btn:
    st.info("👈 Sube un archivo .zip, mapea tus columnas en la barra lateral y presiona **Procesar y Analizar**.")
    st.stop()

if not uploaded_zip or not col_sig:
    st.warning("⚠️ Faltan datos o no se seleccionó la columna de señal.")
    st.stop()

# Procesamiento de datos
with st.spinner("Descomprimiendo y extrayendo indicadores RMS/Kurtosis..."):
    try:
        # Pasamos los bytes del ZIP para evitar problemas con la caché de Streamlit
        df = process_zip_files(uploaded_zip.getvalue(), sep_val, col_time, col_sig)
    except Exception as e:
        st.error(f"Error al procesar los archivos: {e}")
        st.stop()

if df.empty: 
    st.warning("No se extrajeron datos. Verifica que seleccionaste la columna correcta y el separador adecuado.")
    st.stop()

# Thresholds
bn = max(2, int(len(df) * baseline_pct / 100))
st.caption(f"📐 Baseline automático: primeros **{bn} archivos** ({baseline_pct}% de {len(df)})")
base_rms  = df["rms"].iloc[:bn];  mu_r, sd_r = base_rms.mean(),  base_rms.std(ddof=1) if bn>1 else 0.
base_kurt = df["kurtosis_excess"].iloc[:bn]; mu_k, sd_k = base_kurt.mean(), base_kurt.std(ddof=1) if bn>1 else 0.
thr_rms  = mu_r + sigma_mult*sd_r
thr_kurt = mu_k + sigma_mult*sd_k
alarm_rms  = df[df["rms"]             > thr_rms]
alarm_kurt = df[df["kurtosis_excess"] > thr_kurt]

# ─────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
def kpi(col,label,value,fmt=".4f"):
    col.markdown(f"""<div class="metric-card"><div class="metric-label">{label}</div>
    <div class="metric-value">{value:{fmt}}</div></div>""", unsafe_allow_html=True)
kpi(k1,"Archivos (Eventos)",len(df),"d")
kpi(k2,"RMS Máx",df["rms"].max())
kpi(k3,"Kurtosis Máx",df["kurtosis_excess"].max(),".2f")
kpi(k4,f"Umbral RMS ({sigma_mult}σ)",thr_rms)
has_alarm = len(alarm_rms)>0 or len(alarm_kurt)>0
lbl = "🔴 ALARMA" if has_alarm else "🟢 NORMAL"; cls = "status-alarm" if has_alarm else "status-ok"
k5.markdown(f"""<div class="metric-card"><div class="metric-label">Estado Condición</div>
<div class="metric-value" style="font-size:1.4rem"><span class="{cls}">{lbl}</span></div></div>""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────
c1,c2 = st.columns(2)
with c1: st.plotly_chart(plotly_line(df,"rms","Tendencia RMS (Energía Global)",PAL["accent"],thr_rms,f"Umbral {sigma_mult:.1f}σ"),use_container_width=True)
with c2: st.plotly_chart(plotly_line(df,"kurtosis_excess","Tendencia Kurtosis (Impulsividad)","#A78BFA",thr_kurt,f"Umbral {sigma_mult:.1f}σ"),use_container_width=True)
st.plotly_chart(plotly_line(df,"crest_factor","Tendencia Factor de Cresta (Picos)","#34D399"),use_container_width=True)

# ─────────────────────────────────────────
# ALARM PANEL
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("### 🚨 Panel de Alarmas y Excepciones")
ca,cb = st.columns(2)
with ca:
    st.markdown(f"**RMS — {len(alarm_rms)} eventos** (umbral = `{thr_rms:.5f}`)")
    if alarm_rms.empty: st.markdown("<span class='status-ok'>✔ Sin alarmas RMS</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='status-alarm'>Primera alarma en: {alarm_rms.iloc[0]['datetime']}</span>", unsafe_allow_html=True)
        with st.expander("Ver detalle de eventos"): st.dataframe(alarm_rms[["datetime","filename","rms"]], use_container_width=True, hide_index=True)
with cb:
    st.markdown(f"**Kurtosis — {len(alarm_kurt)} eventos** (umbral = `{thr_kurt:.4f}`)")
    if alarm_kurt.empty: st.markdown("<span class='status-ok'>✔ Sin alarmas Kurtosis</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='status-alarm'>Primera alarma en: {alarm_kurt.iloc[0]['datetime']}</span>", unsafe_allow_html=True)
        with st.expander("Ver detalle de eventos"): st.dataframe(alarm_kurt[["datetime","filename","kurtosis_excess"]], use_container_width=True, hide_index=True)

# ─────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────
st.markdown("---")
e1,e2 = st.columns(2)
with e1:
    st.download_button("⬇️  Descargar Histórico CSV", data=df.to_csv(index=False).encode(), file_name="tendencia_vibraciones.csv", mime="text/csv", use_container_width=True)
with e2:
    with st.spinner("Compilando reporte técnico PDF..."):
        pdf_bytes = generate_pdf_report(
            df=df, thr_rms=thr_rms, thr_kurt=thr_kurt, sigma_mult=sigma_mult, baseline_n=bn,
            alarm_rms=alarm_rms, alarm_kurt=alarm_kurt, client_name=client_name, equipo=equipo_name, ingeniero=ingeniero_nm, fs_hz=fs_hz
        )
    st.download_button("📄  Descargar Informe Técnico PDF", data=pdf_bytes, file_name=f"Reporte_Vibraciones_{equipo_name.replace(' ','_')}.pdf", mime="application/pdf", use_container_width=True)
