import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from preguntas import (data_2020, data_2022, level_map) 

TABLAS_GRAFICOS_DIR = "Tablas_y_Graficos_DANE_v3" 
if not os.path.exists(TABLAS_GRAFICOS_DIR):
    os.makedirs(TABLAS_GRAFICOS_DIR)

ING_COL_NAME = 'ingtot'
EDU_NIVEL_2020_NAME = 'p3042'
EDU_NIVEL_2022_NAME = 'p6210'
SEX_2020_NAME = 'p3271'
SEX_2022_NAME = 'p6020'
HRS_COL_NAME = 'p6800'
EDU_ANOS_2020_NAME = 'p3042s1'
EDU_ANOS_2022_NAME = 'p6210s1'


nombres_variables_dane = {
    ING_COL_NAME: "Ingreso Total Mensual",
    EDU_NIVEL_2020_NAME: "Nivel Educativo Alcanzado (GEIH 2020)",
    EDU_NIVEL_2022_NAME: "Nivel Educativo Alcanzado (GEIH 2022)",
    EDU_ANOS_2020_NAME: "Años Totales de Estudio (GEIH 2020)", # Añadido
    EDU_ANOS_2022_NAME: "Años Totales de Estudio (GEIH 2022)", # Añadido
    SEX_2020_NAME: "Sexo (p3271, GEIH 2020)",
    SEX_2022_NAME: "Sexo (p6020, GEIH 2022)",
    HRS_COL_NAME: "Horas Trabajadas Semanales",
}

def obtener_nombre_descriptivo(col_name_actual):
    return nombres_variables_dane.get(col_name_actual, col_name_actual.replace('_', ' ').title())

def guardar_tabla_como_imagen(df, filename, titulo_base, anio_fuente="", col_width_multiplier=2.0, row_height_multiplier=0.6):
    df_to_plot = df.copy()
    if df_to_plot.index.name:
        df_to_plot.reset_index(inplace=True)
    fig_width = len(df_to_plot.columns) * col_width_multiplier
    fig_height = (len(df_to_plot) + 1) * row_height_multiplier + (1.5 if titulo_base else 0.5)
    fig_height = max(fig_height, 4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight'); ax.axis('off')
    full_titulo = titulo_base
    if anio_fuente:
        full_titulo += f"\nFuente: GEIH {anio_fuente} - DANE"
    fig.suptitle(full_titulo, fontsize=14, fontweight='bold', y=0.98 if len(df_to_plot)<5 else 0.96, va='top')
    tabla = ax.table(cellText=df_to_plot.values, colLabels=df_to_plot.columns,
                     cellLoc='center', loc='center', colColours=['#D8BFD8'] * len(df_to_plot.columns))
    tabla.auto_set_font_size(False); tabla.set_fontsize(9); tabla.scale(1, 1.6)
    for key, cell in tabla.get_celld().items():
        cell.set_edgecolor('darkgrey')
        if key[0] == 0:
            cell.set_text_props(weight='semibold', color='black'); cell.set_facecolor('#D8BFD8')
        else:
            cell.set_facecolor('white')
            if key[1] == 0 and isinstance(df_to_plot.iloc[key[0]-1, key[1]], str):
                cell.set_text_props(ha='left', va='center'); cell.PAD = 0.05
    filepath = os.path.join(TABLAS_GRAFICOS_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300, pad_inches=0.2)
    plt.close(fig)
    print(f"Tabla guardada: {filepath}")

def graficar_regresion(df, x_col, y_col, modelo, filename, anio_fuente):
    x_nombre_desc = obtener_nombre_descriptivo(x_col)
    y_nombre_desc = obtener_nombre_descriptivo(y_col)
    titulo_reg = f"Regresión Lineal: {y_nombre_desc} vs. {x_nombre_desc}\nFuente: GEIH {anio_fuente} - DANE"
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_col, y=y_col, data=df, alpha=0.3, color="dodgerblue", s=40, edgecolor='w', linewidth=0.5)
    
    unique_x = df[x_col].dropna().unique()
    if len(unique_x) == 0:
        print(f"No hay valores únicos para X ({x_col}) en la regresión después de dropna. No se puede graficar la línea.")
        plt.close()
        return

    x_pred_vals = np.array(sorted(unique_x))
    X_plot = sm.add_constant(x_pred_vals)
    y_pred = modelo.predict(X_plot)
    
    intercepto = modelo.params.get('const', 0)
    pendiente = modelo.params.get(x_col, modelo.params[1] if len(modelo.params) > 1 else 0)

    r_cuadrado = modelo.rsquared
    eq_label = f'y = {pendiente:.2f}x + {intercepto:.0f}\n$R^2 = {r_cuadrado:.3f}$'
    plt.plot(x_pred_vals, y_pred, color='crimson', linewidth=2.5, label=eq_label)
    plt.title(titulo_reg, fontsize=15, fontweight='bold')
    plt.xlabel(x_nombre_desc, fontsize=12); plt.ylabel(y_nombre_desc, fontsize=12)
    plt.legend(fontsize=10, loc='best'); plt.grid(True, linestyle=':', alpha=0.6)
    filepath = os.path.join(TABLAS_GRAFICOS_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Gráfico de regresión guardado: {filepath}")

def tabla_estadisticos_descriptivos(df, col_nivel_edu_actual, col_ingreso_actual, anio):
    df_copy = df.copy()
    df_copy[col_nivel_edu_actual] = pd.to_numeric(df_copy[col_nivel_edu_actual], errors='coerce')
    df_copy[col_ingreso_actual] = pd.to_numeric(df_copy[col_ingreso_actual], errors='coerce')
    df_copy.dropna(subset=[col_nivel_edu_actual, col_ingreso_actual], inplace=True)
    
    df_copy['Nivel_Educativo_Nombre'] = df_copy[col_nivel_edu_actual].map(
        lambda x: level_map.get(int(x) if pd.notna(x) and isinstance(x, (int, float)) and str(float(x)).split('.')[0].isdigit() else x, f"Cod_{x}")
    )
    
    #stats 
    tabla_num = df_copy.groupby('Nivel_Educativo_Nombre')[col_ingreso_actual].agg(
        Media_calc='mean',
        Desviacion_calc='std',
        Mediana_calc='median',
        P25_calc=lambda x: x.quantile(0.25),
        P75_calc=lambda x: x.quantile(0.75),
        Observaciones_calc='count'
    )

    #format
    tabla_formateada = pd.DataFrame(index=tabla_num.index)
    if not tabla_num.empty: # Solo formatear si hay datos
        tabla_formateada['Media'] = tabla_num['Media_calc'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
        tabla_formateada['Desviación_Estándar'] = tabla_num['Desviacion_calc'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
        tabla_formateada['Mediana'] = tabla_num['Mediana_calc'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
        tabla_formateada['Percentil_25'] = tabla_num['P25_calc'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
        tabla_formateada['Percentil_75'] = tabla_num['P75_calc'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
        tabla_formateada['Observaciones'] = tabla_num['Observaciones_calc'].astype(int) # El conteo es inherentemente int
    
    tabla_formateada.index.name = obtener_nombre_descriptivo(col_nivel_edu_actual)
    return tabla_formateada


def tabla_contingencia_sexo_educacion(df, col_nivel_edu_actual, col_sexo_actual, anio):
    df_copy = df.copy()
    df_copy[col_nivel_edu_actual] = pd.to_numeric(df_copy[col_nivel_edu_actual], errors='coerce')
    df_copy[col_sexo_actual] = pd.to_numeric(df_copy[col_sexo_actual], errors='coerce')
    df_copy.dropna(subset=[col_nivel_edu_actual, col_sexo_actual], inplace=True)
    
    df_copy['Nivel_Educativo_Desc'] = df_copy[col_nivel_edu_actual].map(
        lambda x: level_map.get(int(x) if pd.notna(x) and isinstance(x, (int, float)) and str(float(x)).split('.')[0].isdigit() else x, f"Cod_{x}")
    )
    df_copy['Sexo_Desc'] = df_copy[col_sexo_actual].map({1.0: 'Hombres (1)', 2.0: 'Mujeres (2)'}).fillna('No Especificado')
    
    tabla = pd.crosstab(df_copy['Nivel_Educativo_Desc'], df_copy['Sexo_Desc'])
    tabla.index.name = obtener_nombre_descriptivo(col_nivel_edu_actual)
    tabla.columns.name = obtener_nombre_descriptivo(col_sexo_actual)
    return tabla

def regresion_simple(df, x_col, y_col):
    df_copy = df.copy()
    df_copy[x_col] = pd.to_numeric(df_copy[x_col], errors='coerce')
    df_copy[y_col] = pd.to_numeric(df_copy[y_col], errors='coerce')
    datos = df_copy[[x_col, y_col]].dropna()
    if datos.empty or len(datos) < 2: return None
    X = sm.add_constant(datos[x_col]); y = datos[y_col]
    return sm.OLS(y, X).fit()

if __name__ == "__main__":
    print(f"--- GENERANDO TABLAS Y GRÁFICOS (Guardados en '{TABLAS_GRAFICOS_DIR}') ---")

    titulo_desc_ing = f"Estadísticos Descriptivos del {obtener_nombre_descriptivo(ING_COL_NAME)}"
    tabla_desc_2020 = tabla_estadisticos_descriptivos(data_2020, EDU_NIVEL_2020_NAME, ING_COL_NAME, 2020)
    guardar_tabla_como_imagen(tabla_desc_2020, "ingreso_por_nivel_educativo_2020.png",
                              f"{titulo_desc_ing} por {obtener_nombre_descriptivo(EDU_NIVEL_2020_NAME)}", anio_fuente="2020")

    tabla_desc_2022 = tabla_estadisticos_descriptivos(data_2022, EDU_NIVEL_2022_NAME, ING_COL_NAME, 2022)
    guardar_tabla_como_imagen(tabla_desc_2022, "ingreso_por_nivel_educativo_2022.png",
                              f"{titulo_desc_ing} por {obtener_nombre_descriptivo(EDU_NIVEL_2022_NAME)}", anio_fuente="2022")

    titulo_cont_sex_edu_2020 = f"Distribución de Personas por {obtener_nombre_descriptivo(SEX_2020_NAME)} y {obtener_nombre_descriptivo(EDU_NIVEL_2020_NAME)}"
    tabla_cont_2020 = tabla_contingencia_sexo_educacion(data_2020, EDU_NIVEL_2020_NAME, SEX_2020_NAME, 2020)
    guardar_tabla_como_imagen(tabla_cont_2020, "contingencia_sexo_vs_nivel_educativo_2020.png",
                              titulo_cont_sex_edu_2020, anio_fuente="2020")

    titulo_cont_sex_edu_2022 = f"Distribución de Personas por {obtener_nombre_descriptivo(SEX_2022_NAME)} y {obtener_nombre_descriptivo(EDU_NIVEL_2022_NAME)}"
    tabla_cont_2022 = tabla_contingencia_sexo_educacion(data_2022, EDU_NIVEL_2022_NAME, SEX_2022_NAME, 2022)
    guardar_tabla_como_imagen(tabla_cont_2022, "contingencia_sexo_vs_nivel_educativo_2022.png",
                              titulo_cont_sex_edu_2022, anio_fuente="2022")
    
    modelo_hrs_ing_2020 = regresion_simple(data_2020, HRS_COL_NAME, ING_COL_NAME)
    if modelo_hrs_ing_2020:
        print(f"\n--- Resumen Regresión: {obtener_nombre_descriptivo(HRS_COL_NAME)} vs. {obtener_nombre_descriptivo(ING_COL_NAME)} (2020) ---")
        print(modelo_hrs_ing_2020.summary())
        df_reg_hrs_2020 = data_2020[[HRS_COL_NAME, ING_COL_NAME]].dropna()
        if not df_reg_hrs_2020.empty:
             graficar_regresion(df_reg_hrs_2020, HRS_COL_NAME, ING_COL_NAME, modelo_hrs_ing_2020,
                               "reg_hrs_vs_ing_2020.png", anio_fuente="2020")

    modelo_hrs_ing_2022 = regresion_simple(data_2022, HRS_COL_NAME, ING_COL_NAME)
    if modelo_hrs_ing_2022:
        print(f"\n--- Resumen Regresión: {obtener_nombre_descriptivo(HRS_COL_NAME)} vs. {obtener_nombre_descriptivo(ING_COL_NAME)} (2022) ---")
        print(modelo_hrs_ing_2022.summary())
        df_reg_hrs_2022 = data_2022[[HRS_COL_NAME, ING_COL_NAME]].dropna()
        if not df_reg_hrs_2022.empty:
            graficar_regresion(df_reg_hrs_2022, HRS_COL_NAME, ING_COL_NAME, modelo_hrs_ing_2022,
                               "reg_hrs_vs_ing_2022.png", anio_fuente="2022")

    # Usar AÑOS de estudio para regresión con Ingreso
    modelo_edu_anos_ing_2020 = regresion_simple(data_2020, EDU_ANOS_2020_NAME, ING_COL_NAME)
    if modelo_edu_anos_ing_2020:
        print(f"\n--- Resumen Regresión: {obtener_nombre_descriptivo(EDU_ANOS_2020_NAME)} vs. {obtener_nombre_descriptivo(ING_COL_NAME)} (2020) ---")
        print(modelo_edu_anos_ing_2020.summary())
        df_reg_edu_2020 = data_2020[[EDU_ANOS_2020_NAME, ING_COL_NAME]].dropna()
        if not df_reg_edu_2020.empty:
            graficar_regresion(df_reg_edu_2020, EDU_ANOS_2020_NAME, ING_COL_NAME, modelo_edu_anos_ing_2020,
                               "reg_educ_anos_vs_ingreso_2020.png", anio_fuente="2020")

    modelo_edu_anos_ing_2022 = regresion_simple(data_2022, EDU_ANOS_2022_NAME, ING_COL_NAME)
    if modelo_edu_anos_ing_2022:
        print(f"\n--- Resumen Regresión: {obtener_nombre_descriptivo(EDU_ANOS_2022_NAME)} vs. {obtener_nombre_descriptivo(ING_COL_NAME)} (2022) ---")
        print(modelo_edu_anos_ing_2022.summary())
        df_reg_edu_2022 = data_2022[[EDU_ANOS_2022_NAME, ING_COL_NAME]].dropna()
        if not df_reg_edu_2022.empty:
            graficar_regresion(df_reg_edu_2022, EDU_ANOS_2022_NAME, ING_COL_NAME, modelo_edu_anos_ing_2022,
                               "reg_educ_anos_vs_ingreso_2022.png", anio_fuente="2022")

    print(f"\n--- FIN: Tablas y gráficos guardados en '{TABLAS_GRAFICOS_DIR}' ---")