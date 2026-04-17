import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog

st.set_page_config(page_title="Optimizador de Logística", layout="wide")

st.title("📦 Sistema Inteligente de Distribución")
st.markdown("Optimización de costos basada en el Método de Transporte.")

# --- 1. CONFIGURACIÓN DE DATOS (Basado en tu imagen) ---
proveedores = ["TAXIS X CERO", "UBER", "LAST MILLIE SA", "ESTAFETA"]
zonas = ["Zona Centro", "Zona Sur", "Zona Norte", "Zona Oriente"]

costos = np.array([
    [42, 56, 35, 48],  # TAXIS X CERO
    [50, 61, 43, 40],  # UBER
    [57, 38, 66, 49],  # LAST MILLIE SA
    [52, 45, 50, 60]   # ESTAFETA
])

oferta_max = [56, 70, 39, 102]
demanda_requerida = [70, 73, 90, 34]

# --- 2. SOLVER: CÁLCULO DE LA SOLUCIÓN ÓPTIMA ---
def resolver_transporte(costos, oferta, demanda):
    num_prov = len(oferta)
    num_zonas = len(demanda)
    c = costos.flatten()
    
    # Restricciones de Oferta (Filas)
    A_eq_oferta = []
    for i in range(num_prov):
        row = np.zeros(num_prov * num_zonas)
        row[i*num_zonas : (i+1)*num_zonas] = 1
        A_eq_oferta.append(row)
        
    # Restricciones de Demanda (Columnas)
    A_eq_demanda = []
    for j in range(num_zonas):
        row = np.zeros(num_prov * num_zonas)
        row[j::num_zonas] = 1
        A_eq_demanda.append(row)
        
    A_eq = np.vstack([A_eq_oferta, A_eq_demanda])
    b_eq = np.concatenate([oferta, demanda])
    
    res = linprog(c, A_eq_high=A_eq, b_eq_high=b_eq, method='highs')
    return res.x.reshape((num_prov, num_zonas)) if res.success else None

# --- 3. INTERFAZ ---
tab1, tab2 = st.tabs(["📊 Solución Sugerida (Óptima)", "⚙️ Configuración de Red"])

with tab2:
    st.subheader("Capacidades Actuales")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Oferta (Paquetes disponibles por Proveedor):**")
        for p, o in zip(proveedores, oferta_max):
            st.write(f"{p}: {o} unidades")
    with col_b:
        st.write("**Demanda (Requerimiento por Zona):**")
        for z, d in zip(zonas, demanda_requerida):
            st.write(f"{z}: {d} unidades")

with tab1:
    solucion = resolver_transporte(costos, oferta_max, demanda_requerida)
    
    if solucion is not None:
        df_optimo = pd.DataFrame(solucion, index=proveedores, columns=zonas)
        costo_minimo = (solucion * costos).sum()
        
        st.success(f"### 🏆 Costo Mínimo Encontrado: ${costo_minimo:,.2f}")
        
        st.write("#### Plan de Distribución Recomendado:")
        st.table(df_optimo.style.format("{:.0f}").background_gradient(cmap="Greens"))
        
        # --- OPCIONES CONVENIENTES (VALOR AGREGADO) ---
        st.divider()
        st.subheader("💡 Recomendaciones Estratégicas")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.info("**Mayor Eficiencia**")
            # Encontrar el proveedor con más carga asignada
            top_prov = df_optimo.sum(axis=1).idxmax()
            st.write(f"Concentrar volumen con **{top_prov}** para negociar tarifas.")
            
        with c2:
            st.warning("**Punto Crítico**")
            # Zona con demanda más alta
            top_zona = df_optimo.sum(axis=0).idxmax()
            st.write(f"La **{top_zona}** satura tu red. Considera un Hub temporal ahí.")
            
        with c3:
            st.error("**Costo de Oportunidad**")
            # Identificar la ruta más cara usada
            mask = solucion > 0
            ruta_cara = costos[mask].max()
            st.write(f"Tu ruta más costosa es de **${ruta_cara}**. Busca proveedores locales adicionales.")
    else:
        st.error("No se encontró una solución factible. Revisa que la Oferta total sea igual a la Demanda total.")

# --- FOOTER ---
st.caption("Nota: El algoritmo utiliza Programación Lineal para minimizar el gasto total de transporte.")

