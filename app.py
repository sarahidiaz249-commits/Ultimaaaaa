import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog

st.set_page_config(page_title="Optimizador de Logística", layout="wide")

st.title("📦 Sistema Inteligente de Distribución")
st.markdown("Optimización de costos basada en el Método de Transporte.")

# --- 1. CONFIGURACIÓN DE DATOS ---
proveedores = ["TAXIS X CERO", "UBER", "LAST MILLIE SA", "ESTAFETA"]
zonas = ["Zona Centro", "Zona Sur", "Zona Norte", "Zona Oriente"]

costos = np.array([
    [42, 56, 35, 48],
    [50, 61, 43, 40],
    [57, 38, 66, 49],
    [52, 45, 50, 60]
])

oferta_max = [56, 70, 39, 102]
demanda_requerida = [70, 73, 90, 34]

# --- 2. SOLVER CORREGIDO ---
def resolver_transporte(costos, oferta, demanda):
    num_prov = len(oferta)
    num_zonas = len(demanda)
    c = costos.flatten()
    
    # Restricciones de Oferta (Filas) -> <=
    A_ub = []
    for i in range(num_prov):
        row = np.zeros(num_prov * num_zonas)
        row[i*num_zonas : (i+1)*num_zonas] = 1
        A_ub.append(row)
        
    # Restricciones de Demanda (Columnas) -> ==
    A_eq = []
    for j in range(num_zonas):
        row = np.zeros(num_prov * num_zonas)
        row[j::num_zonas] = 1
        A_eq.append(row)
        
    res = linprog(c, A_ub=A_ub, b_ub=oferta, A_eq=A_eq, b_eq=demanda, method='highs')
    return res.x.reshape((num_prov, num_zonas)) if res.success else None

# --- 3. INTERFAZ ---
tab1, tab2 = st.tabs(["📊 Solución Sugerida (Óptima)", "⚙️ Configuración de Red"])

with tab2:
    st.subheader("Capacidades Actuales")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Oferta (Proveedores):**")
        for p, o in zip(proveedores, oferta_max):
            st.write(f"{p}: {o} unidades")
    with col_b:
        st.write("**Demanda (Zonas):**")
        for z, d in zip(zonas, demanda_requerida):
            st.write(f"{z}: {d} unidades")

with tab1:
    solucion = resolver_transporte(costos, oferta_max, demanda_requerida)
    
    if solucion is not None:
        # Redondear para evitar decimales extraños por precisión matemática
        solucion = np.round(solucion, 0)
        df_optimo = pd.DataFrame(solucion, index=proveedores, columns=zonas)
        costo_minimo = (solucion * costos).sum()
        
        st.success(f"### 🏆 Costo Mínimo Encontrado: ${costo_minimo:,.2f}")
        st.table(df_optimo.style.format("{:.0f}").background_gradient(cmap="Greens"))
        
        st.divider()
        st.subheader("💡 Recomendaciones Estratégicas")
        c1, c2, c3 = st.columns(3)
        with c1:
            top_prov = df_optimo.sum(axis=1).idxmax()
            st.info(f"**Mayor Eficiencia:**\nConcentrar volumen con **{top_prov}**.")
        with c2:
            top_zona = df_optimo.sum(axis=0).idxmax()
            st.warning(f"**Punto Crítico:**\nLa **{top_zona}** satura tu red.")
        with c3:
            mask = solucion > 0
            ruta_cara = costos[mask].max()
            st.error(f"**Costo Alto:**\nTu ruta más cara es de **${ruta_cara}**.")
    else:
        st.error("No se encontró una solución. Verifica que la Oferta Total sea >= a la Demanda Total.")
