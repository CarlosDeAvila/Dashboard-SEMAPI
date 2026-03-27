from __future__ import annotations

import io
import re
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
# SEMAPI BRAND COLORS
# ─────────────────────────────────────────
SEMAPI_BLUE   = colors.HexColor("#1B3A6B")
SEMAPI_ORANGE = colors.HexColor("#E8720C")
SEMAPI_LGRAY  = colors.HexColor("#F0F4FA")
SEMAPI_DGRAY  = colors.HexColor("#4A4A4A")
WHITE         = colors.white

# ─────────────────────────────────────────
# DASHBOARD PALETTE
# ─────────────────────────────────────────
PAL = {
    "bg": "#0F1117", "card": "#1C1F2E", "accent": "#4F8EF7",
    "danger": "#FF4B4B", "ok": "#00C48C", "text": "#E0E6F0", "muted": "#6B7280",
}
FS_HZ = 20_000
FNAME_DT_RE = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})$")

# ─────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────
st.set_page_config(page_title="IMS Bearing Monitor", page_icon="⚙️",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(f"""
<style>
  html, body, [class*="css"] {{
    background-color:{PAL['bg']}; color:{PAL['text']}; font-family:'Inter',sans-serif;
  }}
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
def parse_dt(fname):
    m = FNAME_DT_RE.match(fname)
    if not m: raise ValueError(fname)
    return datetime(*map(int, m.groups()))

def safe_kurtosis(x):
    x = x.astype(np.float64); xc = x - x.mean(); var = np.mean(xc**2)
    return float(np.mean(xc**4)/var**2 - 3.) if var > 0 else np.nan

def compute_features(signal):
    s = signal.astype(np.float64)
    rms = float(np.sqrt(np.mean(s**2))); peak = float(np.max(np.abs(s)))
    kurt = safe_kurtosis(s); cf = peak/rms if rms > 0 else np.nan
    return {"rms": rms, "peak": peak, "kurtosis_excess": kurt, "crest_factor": cf}

@st.cache_data(show_spinner=False)
def process_uploaded_files(files):
    rows = []
    # Ordenamos los archivos por nombre
    sorted_files = sorted(files, key=lambda f: parse_dt(f.name))
    
    for f in sorted_files:
        dt = parse_dt(f.name)
        # Pandas puede leer directamente el objeto del archivo subido
        sig = pd.read_csv(f, header=None, sep=r"\s+", engine="python").iloc[:,0].astype(np.float64).to_numpy()
        feats = compute_features(sig)
        rows.append({"datetime": dt, "filename": f.name, "n_samples": sig.size, **feats})
        
    return pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)

# ─────────────────────────────────────────
# PLOTLY CHARTS
# ─────────────────────────────────────────
def plotly_line(df, y_col, title, color, threshold=None, thr_label="Umbral"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["datetime"], y=df[y_col], mode="lines", name=y_col,
                             line=dict(color=color, width=1.8),
                             hovertemplate="<b>%{x}</b><br>"+y_col+": %{y:.4f}<extra></extra>"))
    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dash", line_color=PAL["danger"],
                      annotation_text=thr_label, annotation_font_color=PAL["danger"])
        mask = df[y_col] > threshold
        if mask.any():
            fig.add_trace(go.Scatter(x=df[mask]["datetime"], y=df[mask][y_col],
                                     mode="markers", marker=dict(color=PAL["danger"],size=4),
                                     name="⚠ Alarma"))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=PAL["text"])),
        paper_bgcolor=PAL["card"], plot_bgcolor=PAL["card"],
        font=dict(color=PAL["text"], size=12),
        xaxis=dict(gridcolor="#2A2D3E", title="Tiempo"),
        yaxis=dict(gridcolor="#2A2D3E"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=20,r=20,t=48,b=20), height=320,
    )
    return fig

# ─────────────────────────────────────────
# PDF HELPERS
# ─────────────────────────────────────────
def make_chart_png(df, y_col, ylabel, color, threshold=None):
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.plot(df["datetime"], df[y_col], color=color, linewidth=1.2)
    if threshold is not None:
        ax.axhline(threshold, color="#CC3333", linestyle="--", linewidth=1,
                   label=f"Umbral {threshold:.4f}")
        mask = df[y_col] > threshold
        if mask.any():
            ax.scatter(df.loc[mask,"datetime"], df.loc[mask,y_col],
                       color="#CC3333", s=12, zorder=5)
        ax.legend(fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8); ax.set_xlabel("Tiempo", fontsize=8)
    ax.tick_params(axis="x", rotation=25, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, alpha=0.3); ax.set_facecolor("#F8F9FB")
    fig.patch.set_facecolor("white"); fig.tight_layout(pad=0.5)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return buf

def semapi_styles():
    S = {}
    S["cover_title"] = ParagraphStyle("ct", fontSize=22, textColor=SEMAPI_BLUE,
        alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=6, leading=28)
    S["cover_sub"]   = ParagraphStyle("cs", fontSize=14, textColor=SEMAPI_BLUE,
        alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=4, leading=18)
    S["cover_body"]  = ParagraphStyle("cb", fontSize=11, textColor=SEMAPI_DGRAY,
        alignment=TA_CENTER, fontName="Helvetica", leading=16)
    S["sec_hdr"]     = ParagraphStyle("sh", fontSize=11, textColor=WHITE,
        fontName="Helvetica-Bold", alignment=TA_LEFT, spaceAfter=0, leading=14)
    S["body"]        = ParagraphStyle("bo", fontSize=9, textColor=SEMAPI_DGRAY,
        fontName="Helvetica", leading=13, alignment=TA_JUSTIFY)
    S["bold_blue"]   = ParagraphStyle("bb", fontSize=9, textColor=SEMAPI_BLUE,
        fontName="Helvetica-Bold", leading=13)
    return S

def _hf(canvas, doc, fecha_full):
    canvas.saveState()
    W, H = letter
    # Header
    canvas.setFillColor(SEMAPI_BLUE); canvas.rect(0, H-50, W, 50, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 15); canvas.drawString(0.4*inch, H-32, "SEMAPI")
    canvas.setFont("Helvetica", 7);       canvas.drawString(0.4*inch, H-43, "Servicios de Mantenimiento e Ingeniería")
    canvas.setFont("Helvetica-Bold", 9);  canvas.drawRightString(W-0.4*inch, H-28, "INFORME DE ANÁLISIS DE VIBRACIONES")
    canvas.setFont("Helvetica", 8);       canvas.drawRightString(W-0.4*inch, H-40, fecha_full)
    # Footer
    canvas.setFillColor(SEMAPI_BLUE); canvas.rect(0, 0, W, 32, fill=1, stroke=0)
    canvas.setFillColor(WHITE); canvas.setFont("Helvetica", 7)
    canvas.drawCentredString(W/2, 18, "Dir.: Via 40 #82 - 47  |  Tels: 3788848 - 3003621 - 3731357  |  Barranquilla")
    canvas.drawCentredString(W/2, 8,  "E-mail: Gerenteservicios@semapicolombia.com / info@semapicolombia.com")
    canvas.drawRightString(W-0.3*inch, 18, f"Pág. {doc.page}")
    canvas.restoreState()

def section_box(label, S):
    tbl = Table([[Paragraph(label, S["sec_hdr"])]], colWidths=[6.9*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),SEMAPI_BLUE),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),8),
    ]))
    return tbl

# ─────────────────────────────────────────
# MAIN PDF GENERATOR
# ─────────────────────────────────────────
def generate_pdf_report(df, thr_rms, thr_kurt, sigma_mult, baseline_n,
                         alarm_rms, alarm_kurt, client_name, equipo, ingeniero):
    buf = io.BytesIO()
    fecha_full = datetime.now().strftime("%d/%m/%Y")
    fecha_str  = datetime.now().strftime("%B %Y").capitalize()
    S = semapi_styles()

    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.6*inch, rightMargin=0.6*inch,
                            topMargin=0.85*inch, bottomMargin=0.65*inch)
    story = []

    # ── PORTADA ──────────────────────────────────
    story.append(Spacer(1, 1.2*inch))
    story.append(Paragraph("INFORME DE", S["cover_title"]))
    story.append(Paragraph("ANÁLISIS DE VIBRACIONES", S["cover_title"]))
    story.append(Spacer(1, 0.4*inch))
    story.append(HRFlowable(width="70%", thickness=2, color=SEMAPI_ORANGE,
                             lineCap="round", spaceAfter=18, spaceBefore=4, hAlign="CENTER"))
    story.append(Paragraph(client_name.upper(), S["cover_sub"]))
    story.append(Spacer(1, 0.25*inch))
    story.append(Paragraph(fecha_str, S["cover_body"]))
    story.append(Spacer(1, 1.2*inch))
    story.append(Paragraph("Dataset IMS – Universidad de Cincinnati<br/>Análisis de degradación de rodamientos", S["cover_body"]))
    story.append(PageBreak())

    # ── INTRO ─────────────────────────────────────
    story.append(section_box("REF.: ANÁLISIS DE VIBRACIONES", S))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Se efectuó servicio de análisis de vibraciones al equipo <b>{equipo}</b>, "
        f"correspondiente al <i>IMS Bearing Dataset</i> de la Universidad de Cincinnati. "
        f"Se procesaron <b>{len(df)}</b> archivos temporales con frecuencia de muestreo de "
        f"{FS_HZ/1000:.0f} kHz, extrayendo indicadores estadísticos en dominio del tiempo.",
        S["body"]
    ))
    story.append(Spacer(1, 14))

    # ── TABLA RESUMEN ─────────────────────────────
    story.append(section_box("TABLA DE DATOS – INDICADORES ESTADÍSTICOS", S))
    story.append(Spacer(1, 6))
    hdr = ["Indicador", "Valor Inicial", "Valor Final", "Valor Máximo", "Umbral alarma"]
    rows = [
        ["RMS (aceleración)",   f"{df['rms'].iloc[0]:.5f}",             f"{df['rms'].iloc[-1]:.5f}",             f"{df['rms'].max():.5f}",             f"{thr_rms:.5f}"],
        ["Kurtosis (excess)",   f"{df['kurtosis_excess'].iloc[0]:.3f}", f"{df['kurtosis_excess'].iloc[-1]:.3f}", f"{df['kurtosis_excess'].max():.3f}", f"{thr_kurt:.3f}"],
        ["Crest Factor",        f"{df['crest_factor'].iloc[0]:.3f}",   f"{df['crest_factor'].iloc[-1]:.3f}",   f"{df['crest_factor'].max():.3f}",   "—"],
        ["Peak (aceleración)",  f"{df['peak'].iloc[0]:.5f}",            f"{df['peak'].iloc[-1]:.5f}",            f"{df['peak'].max():.5f}",            "—"],
    ]
    ft = Table([hdr]+rows, colWidths=[1.9*inch,1.0*inch,1.0*inch,1.0*inch,1.1*inch], repeatRows=1)
    ft.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),SEMAPI_BLUE),("TEXTCOLOR",(0,0),(-1,0),WHITE),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),("ALIGN",(0,0),(0,-1),"LEFT"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[SEMAPI_LGRAY,WHITE]),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CCCCCC")),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("LEFTPADDING",(0,0),(0,-1),6),
    ]))
    story.append(ft)
    story.append(Spacer(1,4))
    story.append(Paragraph(f"<i>Baseline: primeras {baseline_n} muestras · Umbral = media + {sigma_mult:.1f}σ</i>", S["body"]))
    story.append(Spacer(1,14))

    # ── GRÁFICAS ─────────────────────────────────
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

    # ── DIAGNÓSTICO ──────────────────────────────
    n_rms  = len(alarm_rms)
    n_kurt = len(alarm_kurt)
    has_alarm = n_rms > 0 or n_kurt > 0

    if has_alarm:
        candidates = []
        if n_rms  > 0: candidates.append(alarm_rms.iloc[0]["datetime"])
        if n_kurt > 0: candidates.append(alarm_kurt.iloc[0]["datetime"])
        first_dt = min(candidates).strftime("%Y-%m-%d %H:%M")
        diag = (
            f"Los niveles de vibración del rodamiento presentan una <b>tendencia creciente</b> "
            f"que supera el umbral estadístico (media + {sigma_mult:.1f}σ sobre las primeras {baseline_n} muestras). "
        )
        if n_rms  > 0: diag += f"El indicador RMS registra <b>{n_rms} eventos</b> sobre el umbral ({thr_rms:.5f}), primer evento: <b>{first_dt}</b>. "
        if n_kurt > 0: diag += f"La Kurtosis registra <b>{n_kurt} eventos</b> de impulsividad anormal, indicativo de posible daño en pistas o elementos rodantes. "
        diag += "Se recomienda complementar el análisis con espectros de frecuencia para identificar frecuencias características de falla (BPFI, BPFO, BSF, FTF)."
        reco = ("Revisar el estado físico del rodamiento. Programar inspección con análisis espectral en alta frecuencia (Envelope/PeakVue). "
                "Lubricar con grasa recomendada por el fabricante y mantener bajo seguimiento vibracional periódico.")
    else:
        diag = (
            f"Los niveles de vibración del rodamiento se encuentran dentro de parámetros satisfactorios "
            f"según el umbral estadístico de referencia (media + {sigma_mult:.1f}σ). "
            f"RMS máximo: <b>{df['rms'].max():.5f}</b> · Kurtosis máxima: <b>{df['kurtosis_excess'].max():.3f}</b>. "
            "No se observan indicios de degradación significativa en el periodo analizado."
        )
        reco = "Mantener bajo seguimiento vibracional mensual. Conservar los registros históricos para comparación en próximas mediciones."

    story.append(KeepTogether([section_box("DIAGNÓSTICO", S), Spacer(1,6), Paragraph(diag, S["body"])]))
    story.append(Spacer(1,12))

    # ── PANEL DE ALARMAS ─────────────────────────
    if has_alarm:
        story.append(section_box("PANEL DE ALARMAS", S))
        story.append(Spacer(1,6))
        ahdr = ["Métrica","N° eventos","Primera alarma","Umbral"]
        arows = []
        if n_rms  > 0: arows.append(["RMS",str(n_rms),alarm_rms.iloc[0]["datetime"].strftime("%Y-%m-%d %H:%M"),f"{thr_rms:.5f}"])
        if n_kurt > 0: arows.append(["Kurtosis (excess)",str(n_kurt),alarm_kurt.iloc[0]["datetime"].strftime("%Y-%m-%d %H:%M"),f"{thr_kurt:.3f}"])
        at = Table([ahdr]+arows, colWidths=[1.8*inch,0.9*inch,2.0*inch,1.4*inch], repeatRows=1)
        at.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),SEMAPI_ORANGE),("TEXTCOLOR",(0,0),(-1,0),WHITE),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),
            ("ALIGN",(1,0),(-1,-1),"CENTER"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#FFF3EB"),WHITE]),
            ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CCCCCC")),
            ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ]))
        story.append(at)
        story.append(Spacer(1,12))

    # ── RECOMENDACIÓN ─────────────────────────────
    story.append(KeepTogether([section_box("RECOMENDACIÓN", S), Spacer(1,6), Paragraph(reco, S["body"])]))
    story.append(Spacer(1,24))

    # ── FIRMAS ───────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=SEMAPI_BLUE, spaceBefore=4, spaceAfter=10))
    story.append(Paragraph("Cualquier comentario o inquietud con gusto lo atenderemos.", S["body"]))
    story.append(Spacer(1,18))
    fdata = [
        [Paragraph("<b>Realizó:</b>", S["bold_blue"]),  Paragraph("<b>Revisó:</b>", S["bold_blue"])],
        [Paragraph(ingeniero, S["body"]),                Paragraph("Ing. Eloy Balza", S["body"])],
        [Paragraph("Líder de Mantenimiento", S["body"]),Paragraph("", S["body"])],
        [Paragraph("Especialista en Vibraciones y Alineación Láser", S["body"]), Paragraph("", S["body"])],
    ]
    ftbl = Table(fdata, colWidths=[3.4*inch, 3.4*inch])
    ftbl.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),3)]))
    story.append(ftbl)

    doc.build(story,
              onFirstPage=lambda c,d: _hf(c,d,fecha_full),
              onLaterPages=lambda c,d: _hf(c,d,fecha_full))
    return buf.getvalue()

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    st.markdown("---")
    
    uploaded_files = st.file_uploader(
        "📥 Sube los archivos del IMS Dataset", 
        accept_multiple_files=True
    )
    
    st.markdown("#### Parámetros de alarma")
    baseline_n  = st.number_input("Muestras baseline", min_value=10, max_value=1000, value=200, step=10)
    sigma_mult  = st.slider("Multiplicador σ", 1.0, 5.0, 3.0, 0.5)
    st.markdown("#### Datos del informe")
    client_name  = st.text_input("Cliente",  value="EMPRESA CLIENTE")
    equipo_name  = st.text_input("Equipo",   value="Rodamiento IMS – Canal 1")
    ingeniero_nm = st.text_input("Realizó",  value="Ing. Donald Florian")
    analyze_btn  = st.button("▶  Analizar", use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown(f"<span style='color:{PAL['muted']};font-size:.78rem'>SEMAPI · IMS Dataset · fs={FS_HZ/1000:.0f} kHz</span>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────
# HEADER & MAIN EXECUTION
# ─────────────────────────────────────────
st.markdown("# ⚙️ IMS Bearing Monitor")
st.markdown("## Análisis de degradación — Mantenimiento Predictivo · SEMAPI")
st.markdown("---")

# Validaciones de inicio
if not analyze_btn:
    st.info("👈 Configura los parámetros, sube tus archivos y presiona **Analizar** para comenzar.")
    st.stop()

if not uploaded_files:
    st.warning("⚠️ Por favor, sube los archivos de datos en la barra lateral para continuar.")
    st.stop()

# Procesamiento de datos
with st.spinner("Cargando y procesando señales..."):
    try:
        df = process_uploaded_files(uploaded_files)
    except Exception as e:
        st.error(f"Error al procesar los archivos: {e}")
        st.stop()

if df.empty: 
    st.warning("No se encontraron archivos válidos o hubo un problema al leerlos.")
    st.stop()

# Thresholds
bn = min(int(baseline_n), len(df))
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
kpi(k1,"Archivos",len(df),"d")
kpi(k2,"RMS Máx",df["rms"].max())
kpi(k3,"Kurtosis Máx",df["kurtosis_excess"].max(),".2f")
kpi(k4,"Umbral RMS",thr_rms)
has_alarm = len(alarm_rms)>0 or len(alarm_kurt)>0
lbl = "🔴 ALARMA" if has_alarm else "🟢 NORMAL"; cls = "status-alarm" if has_alarm else "status-ok"
k5.markdown(f"""<div class="metric-card"><div class="metric-label">Estado</div>
<div class="metric-value" style="font-size:1.4rem"><span class="{cls}">{lbl}</span></div></div>""",
unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────
c1,c2 = st.columns(2)
with c1:
    st.plotly_chart(plotly_line(df,"rms","RMS vs Tiempo",PAL["accent"],thr_rms,f"Umbral {sigma_mult:.1f}σ"),use_container_width=True)
with c2:
    st.plotly_chart(plotly_line(df,"kurtosis_excess","Kurtosis vs Tiempo","#A78BFA",thr_kurt,f"Umbral {sigma_mult:.1f}σ"),use_container_width=True)
st.plotly_chart(plotly_line(df,"crest_factor","Crest Factor vs Tiempo","#34D399"),use_container_width=True)

# ─────────────────────────────────────────
# ALARM PANEL
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("### 🚨 Panel de Alarmas")
ca,cb = st.columns(2)
with ca:
    st.markdown(f"**RMS — {len(alarm_rms)} eventos** (umbral = `{thr_rms:.5f}`)")
    if alarm_rms.empty:
        st.markdown("<span class='status-ok'>✔ Sin alarmas RMS</span>", unsafe_allow_html=True)
    else:
        f0 = alarm_rms.iloc[0]
        st.markdown(f"<span class='status-alarm'>Primera alarma: {f0['datetime']} — {f0['filename']}</span>", unsafe_allow_html=True)
        with st.expander("Ver todos"):
            st.dataframe(alarm_rms[["datetime","filename","rms"]], use_container_width=True, hide_index=True)
with cb:
    st.markdown(f"**Kurtosis — {len(alarm_kurt)} eventos** (umbral = `{thr_kurt:.4f}`)")
    if alarm_kurt.empty:
        st.markdown("<span class='status-ok'>✔ Sin alarmas Kurtosis</span>", unsafe_allow_html=True)
    else:
        fk = alarm_kurt.iloc[0]
        st.markdown(f"<span class='status-alarm'>Primera alarma: {fk['datetime']} — {fk['filename']}</span>", unsafe_allow_html=True)
        with st.expander("Ver todos"):
            st.dataframe(alarm_kurt[["datetime","filename","kurtosis_excess"]], use_container_width=True, hide_index=True)

# ─────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────
st.markdown("---")
e1,e2 = st.columns(2)
with e1:
    st.download_button("⬇️  Descargar CSV", data=df.to_csv(index=False).encode(),
                       file_name="ims_features.csv", mime="text/csv",
                       use_container_width=True)
with e2:
    with st.spinner("Generando informe PDF..."):
        pdf_bytes = generate_pdf_report(
            df=df, thr_rms=thr_rms, thr_kurt=thr_kurt,
            sigma_mult=sigma_mult, baseline_n=bn,
            alarm_rms=alarm_rms, alarm_kurt=alarm_kurt,
            client_name=client_name, equipo=equipo_name, ingeniero=ingeniero_nm,
        )
    st.download_button(
        "📄  Descargar Informe PDF",
        data=pdf_bytes,
        file_name=f"Informe_Vibraciones_{client_name.replace(' ','_')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )