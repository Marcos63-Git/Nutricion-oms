import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import io

# Importaciones para PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Configuración de página
st.set_page_config(
    page_title="Sistema Integral de Evaluación Nutricional OMS/OPS",
    page_icon="👶",
    layout="wide"
)

st.title("👶 Sistema Integral de Evaluación Nutricional Infantil (OMS/OPS)")
st.markdown("Evaluación antropométrica avanzada, seguimiento longitudinal e informes clínicos automatizados.")

# --- Inicializar Base de Datos en Memoria (Session State para Historial) ---
if 'historial' not in st.session_state:
    st.session_state.historial = []

# --- Tablas LMS de Referencia (Base ampliada y estructurada) ---
OMS_LMS = {
    'niños': {
        'talla_edad': pd.DataFrame({
            'meses': [0, 6, 12, 18, 24, 36, 48, 60],
            'L': [1.0]*8,
            'M': [49.9, 67.6, 75.7, 82.3, 87.8, 96.1, 103.3, 110.0],
            'S': [0.038, 0.035, 0.034, 0.034, 0.035, 0.037, 0.038, 0.039]
        }),
        'peso_edad': pd.DataFrame({
            'meses': [0, 6, 12, 18, 24, 36, 48, 60],
            'L': [0.35, 0.12, 0.02, -0.04, -0.08, -0.14, -0.19, -0.23],
            'M': [3.3, 7.9, 9.6, 10.9, 12.2, 14.3, 16.3, 18.3],
            'S': [0.133, 0.111, 0.108, 0.108, 0.110, 0.115, 0.122, 0.129]
        }),
        'imc_edad': pd.DataFrame({
            'meses': [0, 6, 12, 18, 24, 36, 48, 60],
            'L': [0.30, -0.10, -0.25, -0.35, -0.40, -0.45, -0.50, -0.52],
            'M': [13.4, 17.3, 16.8, 16.1, 15.7, 15.3, 15.2, 15.3],
            'S': [0.092, 0.080, 0.078, 0.077, 0.077, 0.078, 0.080, 0.082]
        }),
        'peso_talla': pd.DataFrame({
            'talla': [50, 60, 70, 80, 90, 100, 110, 120],
            'L': [-0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0],
            'M': [3.4, 5.8, 8.5, 10.6, 12.9, 15.6, 18.7, 22.4],
            'S': [0.091, 0.085, 0.082, 0.081, 0.081, 0.083, 0.086, 0.090]
        })
    },
    'niñas': {
        'talla_edad': pd.DataFrame({
            'meses': [0, 6, 12, 18, 24, 36, 48, 60],
            'L': [1.0]*8,
            'M': [49.1, 65.7, 74.0, 80.7, 86.4, 95.1, 102.7, 109.4],
            'S': [0.038, 0.035, 0.034, 0.034, 0.036, 0.038, 0.039, 0.040]
        }),
        'peso_edad': pd.DataFrame({
            'meses': [0, 6, 12, 18, 24, 36, 48, 60],
            'L': [0.38, 0.14, 0.01, -0.06, -0.11, -0.18, -0.23, -0.27],
            'M': [3.2, 7.3, 8.9, 10.2, 11.5, 13.9, 16.1, 18.2],
            'S': [0.140, 0.115, 0.113, 0.113, 0.115, 0.122, 0.130, 0.139]
        }),
        'imc_edad': pd.DataFrame({
            'meses': [0, 6, 12, 18, 24, 36, 48, 60],
            'L': [0.33, -0.08, -0.23, -0.33, -0.38, -0.43, -0.48, -0.50],
            'M': [13.3, 16.9, 16.3, 15.7, 15.3, 15.0, 15.0, 15.2],
            'S': [0.095, 0.084, 0.081, 0.080, 0.080, 0.082, 0.085, 0.088]
        }),
        'peso_talla': pd.DataFrame({
            'talla': [50, 60, 70, 80, 90, 100, 110, 120],
            'L': [-0.38, -0.32, -0.27, -0.22, -0.17, -0.12, -0.07, -0.02],
            'M': [3.3, 5.4, 8.0, 10.2, 12.5, 15.2, 18.3, 22.0],
            'S': [0.095, 0.088, 0.084, 0.082, 0.083, 0.085, 0.088, 0.092]
        })
    }
}

# --- Funciones de Cálculo ---
def calcular_zscore(y, L, M, S):
    if L != 0:
        return ((y / M) ** L - 1) / (L * S)
    else:
        return np.log(y / M) / S

def obtener_lms_interpolado(df, valor, col_eje='meses'):
    L = np.interp(valor, df[col_eje], df['L'])
    M = np.interp(valor, df[col_eje], df['M'])
    S = np.interp(valor, df[col_eje], df['S'])
    return L, M, S

def diagnostico_zscore(z, indicador):
    if indicador == 'T/E':
        if z < -3: return "Talla Baja Severa 🔴"
        if z < -2: return "Talla Baja 🟠"
        if z > 3: return "Talla Alta 🔵"
        return "Talla Adecuada 🟢"
    elif indicador == 'P/E':
        if z < -3: return "Bajo Peso Severo 🔴"
        if z < -2: return "Bajo Peso 🟠"
        if z > 2: return "Peso Elevado 🟠"
        return "Peso Adecuado 🟢"
    elif indicador in ['P/T', 'IMC/E']:
        if z < -3: return "Emaciación Severa 🔴"
        if z < -2: return "Emaciación 🟠"
        if z > 3: return "Obesidad 🔴"
        if z > 2: return "Sobrepeso 🟠"
        if z > 1: return "Riesgo de Sobrepeso 🟡"
        return "Eutrófico (Normal) 🟢"

# --- Interfaz de Entrada (Sidebar) ---
st.sidebar.header("📋 Datos del Paciente")
nombre = st.sidebar.text_input("Nombre del Paciente", "Mateo Pérez")
sexo = st.sidebar.radio("Sexo", ["Niños", "Niñas"]).lower()

fecha_nacimiento = st.sidebar.date_input("Fecha de Nacimiento", date(2023, 1, 10))
fecha_evaluacion = st.sidebar.date_input("Fecha de Evaluación", date.today())

peso = st.sidebar.number_input("Peso (kg)", min_value=1.0, max_value=50.0, value=12.0, step=0.1)
talla = st.sidebar.number_input("Talla / Longitud (cm)", min_value=45.0, max_value=120.0, value=85.0, step=0.5)

# --- Botón para Registrar Visita ---
registrar = st.sidebar.button("💾 Guardar Visita en Historial")

# --- Procesamiento ---
dias_edad = (fecha_evaluacion - fecha_nacimiento).days
meses_edad = dias_edad / 30.4375
imc = peso / ((talla / 100) ** 2)

if dias_edad < 0:
    st.error("Error: La fecha de evaluación no puede ser anterior al nacimiento.")
else:
    datos_sexo = OMS_LMS[sexo]
    
    L_te, M_te, S_te = obtener_lms_interpolado(datos_sexo['talla_edad'], meses_edad, 'meses')
    z_te = calcular_zscore(talla, L_te, M_te, S_te)

    L_pe, M_pe, S_pe = obtener_lms_interpolado(datos_sexo['peso_edad'], meses_edad, 'meses')
    z_pe = calcular_zscore(peso, L_pe, M_pe, S_pe)

    L_imc, M_imc, S_imc = obtener_lms_interpolado(datos_sexo['imc_edad'], meses_edad, 'meses')
    z_imc = calcular_zscore(imc, L_imc, M_imc, S_imc)

    L_pt, M_pt, S_pt = obtener_lms_interpolado(datos_sexo['peso_talla'], talla, 'talla')
    z_pt = calcular_zscore(peso, L_pt, M_pt, S_pt)

    if registrar:
        nueva_visita = {
            "Fecha": fecha_evaluacion,
            "Edad (meses)": round(meses_edad, 1),
            "Peso (kg)": peso,
            "Talla (cm)": talla,
            "Z-T/E": round(z_te, 2),
            "Z-P/E": round(z_pe, 2),
            "Z-IMC/E": round(z_imc, 2)
        }
        st.session_state.historial.append(nueva_visita)
        st.sidebar.success("¡Visita guardada con éxito en el historial!")

    # --- Panel Principal de Resultados ---
    st.subheader(f"Resultados Clínicos Actuales: {nombre}")
    st.info(f"**Edad:** {dias_edad // 365} años, {(dias_edad % 365) // 30} meses ({meses_edad:.1f} meses totales) | **IMC:** {imc:.2f} kg/m²")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Talla / Edad", f"{z_te:+.2f} SD")
        st.write(f"{diagnostico_zscore(z_te, 'T/E')}")
    with col2:
        st.metric("Peso / Edad", f"{z_pe:+.2f} SD")
        st.write(f"{diagnostico_zscore(z_pe, 'P/E')}")
    with col3:
        st.metric("IMC / Edad", f"{z_imc:+.2f} SD")
        st.write(f"{diagnostico_zscore(z_imc, 'IMC/E')}")
    with col4:
        st.metric("Peso / Talla", f"{z_pt:+.2f} SD")
        st.write(f"{diagnostico_zscore(z_pt, 'P/T')}")

    # --- Historial Longitudinal ---
    if len(st.session_state.historial) > 0:
        st.markdown("---")
        st.subheader("📈 Historial de Evolución Longitudinal del Paciente")
        df_hist = pd.DataFrame(st.session_state.historial)
        st.dataframe(df_hist, use_container_width=True)

    # --- Generación de Reporte PDF con ReportLab ---
    st.markdown("---")
    st.subheader("📄 Generación de Informe Clínico Oficial")

    def generar_pdf():
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elementos = []
        
        estilos = getSampleStyleSheet()
        estilo_titulo = ParagraphStyle('TituloClinico', parent=estilos['Heading1'], fontSize=16, textColor=colors.HexColor("#1f4e78"))
        estilo_sub = ParagraphStyle('SubClinico', parent=estilos['Heading2'], fontSize=12, textColor=colors.HexColor("#333333"))
        estilo_normal = estilos['Normal']

        elementos.append(Paragraph("INFORME DE EVALUACIÓN NUTRICIONAL INFANTIL (OMS/OPS)", estilo_titulo))
        elementos.append(Spacer(1, 10))
        
        datos_paciente = [
            [Paragraph(f"<b>Paciente:</b> {nombre}", estilo_normal), Paragraph(f"<b>Sexo:</b> {sexo.capitalize()}", estilo_normal)],
            [Paragraph(f"<b>F. Nacimiento:</b> {fecha_nacimiento}", estilo_normal), Paragraph(f"<b>F. Evaluación:</b> {fecha_evaluacion}", estilo_normal)],
            [Paragraph(f"<b>Edad Exacta:</b> {meses_edad:.1f} meses", estilo_normal), Paragraph(f"<b>Antropometría:</b> {peso} kg / {talla} cm", estilo_normal)]
        ]
        t_paciente = Table(datos_paciente, colWidths=[200, 200])
        t_paciente.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f2f2f2")), ('PADDING', (0,0), (-1,-1), 6)]))
        elementos.append(t_paciente)
        elementos.append(Spacer(1, 15))

        elementos.append(Paragraph("Diagnóstico Antropométrico por Z-Score", estilo_sub))
        elementos.append(Spacer(1, 5))

        datos_tabla = [
            ["Indicador", "Puntuación Z", "Diagnóstico OMS"],
            ["Talla para la Edad (T/E)", f"{z_te:+.2f} SD", diagnostico_zscore(z_te, 'T/E')],
            ["Peso para la Edad (P/E)", f"{z_pe:+.2f} SD", diagnostico_zscore(z_pe, 'P/E')],
            ["IMC para la Edad (IMC/E)", f"{z_imc:+.2f} SD", diagnostico_zscore(z_imc, 'IMC/E')],
            ["Peso para la Talla (P/T)", f"{z_pt:+.2f} SD", diagnostico_zscore(z_pt, 'P/T')],
        ]
        t_diag = Table(datos_tabla, colWidths=[150, 90, 220])
        t_diag.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1f4e78")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('PADDING', (0,0), (-1,-1), 6)
        ]))
        elementos.append(t_diag)
        doc.build(elementos)
        buffer.seek(0)
        return buffer
    # --- Gráficos de Curvas de Crecimiento OMS ---
    st.subheader("Curvas de Crecimiento OMS")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    # Aquí configuras el ploteo de las curvas y los puntos de tu paciente
    ax.set_title("Seguimiento Antropométrico")
    ax.set_xlabel("Edad (meses)")
    ax.set_ylabel("Puntuación Z / Medida")
    ax.grid(True)
    
    # Renderizamos la figura en la interfaz de Streamlit
    st.pyplot(fig)

    pdf_file = generar_pdf()
    st.download_button(
        label="📥 Descargar Informe Clínico en PDF",
        data=pdf_file,
        file_name=f"Evaluacion_Nutricional_{nombre.replace(' ', '_')}.pdf",
        mime="application/pdf"
    )
