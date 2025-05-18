import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
import os

GRAFICOS_OUTPUT_DIR = "TODOS_LOS_GRAFICOS_DANE_v3"
if not os.path.exists(GRAFICOS_OUTPUT_DIR):
    os.makedirs(GRAFICOS_OUTPUT_DIR)

import preguntas as po
data_2020 = po.data_2020.copy()
data_2022 = po.data_2022.copy()
level_map_dict = po.level_map
pairs_list = po.pairs
T_CRITICO_val = po.T_CRITICO
Z_CRITICO_val = po.Z_CRITICO
F_LOWER_val = po.F_LOWER
F_UPPER_val = po.F_UPPER

ING_COL = 'ingtot'
EDU_2020_NIVEL_COL = 'p3042'
EDU_2022_NIVEL_COL = 'p6210'
EDU_2020_ANOS_COL = 'p3042s1'
EDU_2022_ANOS_COL = 'p6210s1'
EDU_COL_P1_2020 = EDU_2020_ANOS_COL 
EDU_COL_P1_2022 = EDU_2022_ANOS_COL
SEX_COL_2020 = 'p3271'
SEX_COL_2022 = 'p6020'
HRS_COL = 'p6800'
EDAD_COL = 'p6040'

colores = {
    "naranja_oscuro": "#D52D00", "naranja": "#EF7627", "naranja_clarito": "#FF9A56",
    "rosado_clarito": "#D162A4", "rosado": "#B55690", "rosado_oscuro": "#A30262",
    "azul_principal": "steelblue", "azul_observado": "blue",
}

nombres_variables_dane = {
    ING_COL: "Ingreso Total Mensual",
    EDU_2020_NIVEL_COL: "Nivel Educativo Alcanzado (GEIH 2020)",
    EDU_2022_NIVEL_COL: "Nivel Educativo Alcanzado (GEIH 2022)",
    EDU_2020_ANOS_COL: "Años Totales de Estudio (GEIH 2020)",
    EDU_2022_ANOS_COL: "Años Totales de Estudio (GEIH 2022)",
    SEX_COL_2020: "Sexo (p3271, GEIH 2020)",
    SEX_COL_2022: "Sexo (p6020, GEIH 2022)",
    HRS_COL: "Horas Trabajadas Semanales",
    EDAD_COL: "Edad en Años Cumplidos"
}
def obtener_nombre_descriptivo(col_name_actual, año_num=None):
    if col_name_actual == SEX_COL_2020 and año_num == 2020:
        return nombres_variables_dane.get(SEX_COL_2020, "Sexo (2020)")
    if col_name_actual == SEX_COL_2022 and año_num == 2022:
        return nombres_variables_dane.get(SEX_COL_2022, "Sexo (2022)")
    return nombres_variables_dane.get(col_name_actual, col_name_actual.replace('_', ' ').title())

def hist_nivel_educativo(df, nivel_edu_col, año_num):
    nombre_desc_edu = obtener_nombre_descriptivo(nivel_edu_col, año_num)
    titulo = f"Distribución: {nombre_desc_edu}\nFuente: GEIH {año_num} - DANE"
    xlabel = nombre_desc_edu
    plt.figure(figsize=(10,6))
    data_to_plot = df[nivel_edu_col].dropna()
    if data_to_plot.empty: plt.close(); return
    sns.histplot(data_to_plot, bins=max(10, int(data_to_plot.nunique()/2)) if data_to_plot.nunique() > 20 else data_to_plot.nunique() +1 , kde=False, color=colores["naranja_clarito"], edgecolor='grey')
    plt.title(titulo, fontsize=15)
    plt.xlabel(xlabel, fontsize=12); plt.ylabel('Frecuencia (Conteo)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    filename = f"hist_{nivel_edu_col}_{año_num}.png"
    plt.savefig(os.path.join(GRAFICOS_OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

def barras_ingreso_promedio(df1, df2, ingreso_col):
    media_2020 = df1[ingreso_col].mean(); media_2022 = df2[ingreso_col].mean()
    nombre_desc_ing = obtener_nombre_descriptivo(ingreso_col)
    titulo = f"{nombre_desc_ing} Promedio: 2020 vs. 2022\nFuente: GEIH - DANE"
    plt.figure(figsize=(7,5))
    bars = plt.bar(['2020', '2022'], [media_2020, media_2022], color=[colores["naranja"], colores["rosado"]])
    plt.title(titulo, fontsize=15)
    plt.ylabel(f"{nombre_desc_ing} Promedio ($)", fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(media_2020, media_2022) , f"${yval:,.0f}", ha='center', va='bottom')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    filename = f"barras_{ingreso_col}_promedio_2020vs2022.png"
    plt.savefig(os.path.join(GRAFICOS_OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

def regresion_y_grafico(df, x_col, y_col, año_str_o_num): # Puede ser '2020' o '2020 (Años Estudio)' o 2020
    año_num_titulo = str(año_str_o_num).split(" ")[0] # Extraer solo el número del año para el título de fuente
    x_nombre_desc = obtener_nombre_descriptivo(x_col, int(año_num_titulo) if año_num_titulo.isdigit() else None)
    y_nombre_desc = obtener_nombre_descriptivo(y_col)
    titulo = f"Regresión Lineal: {y_nombre_desc} vs. {x_nombre_desc}\nFuente: GEIH {año_str_o_num} - DANE"
    datos = df[[x_col, y_col]].copy()
    datos[x_col] = pd.to_numeric(datos[x_col], errors='coerce')
    datos[y_col] = pd.to_numeric(datos[y_col], errors='coerce')
    datos.dropna(inplace=True)
    if datos.empty or len(datos) < 2: return None
    X_model = sm.add_constant(datos[x_col]); modelo = sm.OLS(datos[y_col], X_model).fit()
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=x_col, y=y_col, data=datos, alpha=0.3, color=colores["rosado_clarito"], s=40)
    x_pred_vals = np.array(sorted(datos[x_col].unique()))
    X_plot = sm.add_constant(x_pred_vals)
    y_pred = modelo.predict(X_plot)
    intercepto = modelo.params.get('const', 0) # Manejar si no hay intercepto (aunque add_constant lo añade)
    pendiente = modelo.params.get(x_col, 0)    # Manejar si la variable no está en params
    r_cuadrado = modelo.rsquared
    eq_label = f'y = {pendiente:.2f}x + {intercepto:.0f}\n$R^2 = {r_cuadrado:.3f}$'
    plt.plot(x_pred_vals, y_pred, color=colores["naranja_oscuro"], linewidth=2.5, label=eq_label)
    plt.title(titulo, fontsize=15)
    plt.xlabel(x_nombre_desc, fontsize=12); plt.ylabel(y_nombre_desc, fontsize=12)
    plt.legend(fontsize=10); plt.grid(True, linestyle=':', alpha=0.7)
    filename = f"reg_{x_col}_vs_{y_col}_{str(año_str_o_num).replace(' ','_').replace('(','').replace(')','')}.png"
    plt.savefig(os.path.join(GRAFICOS_OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    return modelo

def display_t_distribution(t_stat, t_crit_val, df, title, filename_base, tail_type='right'):
    if df <= 0: return
    t_crit_abs = abs(t_crit_val)
    x_min_ppf, x_max_ppf = po.stats.t.ppf(0.00001, df), po.stats.t.ppf(0.99999, df)
    plot_min_x = min(x_min_ppf, t_stat - 1.5, -t_crit_abs - 1.5)
    plot_max_x = max(x_max_ppf, t_stat + 1.5, t_crit_abs + 1.5)
    x = np.linspace(plot_min_x, plot_max_x, 500); y = po.stats.t.pdf(x, df)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f'Distribución t (df={df:.2f})', color=colores["azul_principal"])
    plt.xlabel("Valor t", fontsize=12); plt.ylabel("Densidad", fontsize=12)
    fill_color = colores["naranja_clarito"]; crit_color = colores["naranja_oscuro"]
    if tail_type == 'right':
        plt.fill_between(x, 0, y, where=(x > t_crit_abs), color=fill_color, alpha=0.7, label=f'Rechazo\nt > {t_crit_abs:.3f}')
        plt.axvline(t_crit_abs, color=crit_color, linestyle='--', label=f'Crítico t = {t_crit_abs:.3f}')
    elif tail_type == 'left':
        plt.fill_between(x, 0, y, where=(x < -t_crit_abs), color=fill_color, alpha=0.7, label=f'Rechazo\nt < {-t_crit_abs:.3f}')
        plt.axvline(-t_crit_abs, color=crit_color, linestyle='--', label=f'Crítico t = {-t_crit_abs:.3f}')
    elif tail_type == 'two':
        plt.fill_between(x, 0, y, where=((x > t_crit_abs) | (x < -t_crit_abs)), color=fill_color, alpha=0.7, label=f'Rechazo\n|t| > {t_crit_abs:.3f}')
        plt.axvline(t_crit_abs, color=crit_color, linestyle='--', label=f'Crítico t = ±{t_crit_abs:.3f}')
        plt.axvline(-t_crit_abs, color=crit_color, linestyle='--')
    plt.axvline(t_stat, color=colores["azul_observado"], linestyle='-', lw=2.5, label=f'Observado t = {t_stat:.3f}')
    plt.title(title, fontsize=15); plt.legend(fontsize=10); plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(os.path.join(GRAFICOS_OUTPUT_DIR, f"{filename_base}_t_dist.png"), dpi=300, bbox_inches='tight')
    plt.close()

def display_z_distribution(z_stat, z_crit_val, title, filename_base, tail_type='right'):
    z_crit_abs = abs(z_crit_val)
    plot_min_x = min(-4.5, z_stat - 1.5, -z_crit_abs - 1.5)
    plot_max_x = max(4.5, z_stat + 1.5, z_crit_abs + 1.5)
    x = np.linspace(plot_min_x, plot_max_x, 500); y = po.stats.norm.pdf(x)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Dist. Normal Estándar', color=colores["azul_principal"])
    plt.xlabel("Valor Z", fontsize=12); plt.ylabel("Densidad", fontsize=12)
    fill_color = colores["rosado_clarito"]; crit_color = colores["rosado_oscuro"]
    if tail_type == 'right':
        plt.fill_between(x, 0, y, where=(x > z_crit_abs), color=fill_color, alpha=0.7, label=f'Rechazo\nZ > {z_crit_abs:.3f}')
        plt.axvline(z_crit_abs, color=crit_color, linestyle='--', label=f'Crítico Z = {z_crit_abs:.3f}')
    elif tail_type == 'left':
        plt.fill_between(x, 0, y, where=(x < -z_crit_abs), color=fill_color, alpha=0.7, label=f'Rechazo\nZ < {-z_crit_abs:.3f}')
        plt.axvline(-z_crit_abs, color=crit_color, linestyle='--', label=f'Crítico Z = {-z_crit_abs:.3f}')
    elif tail_type == 'two':
        plt.fill_between(x, 0, y, where=((x > z_crit_abs) | (x < -z_crit_abs)), color=fill_color, alpha=0.7, label=f'Rechazo\n|Z| > {z_crit_abs:.3f}')
        plt.axvline(z_crit_abs, color=crit_color, linestyle='--', label=f'Crítico Z = ±{z_crit_abs:.3f}')
        plt.axvline(-z_crit_abs, color=crit_color, linestyle='--')
    plt.axvline(z_stat, color=colores["azul_observado"], linestyle='-', lw=2.5, label=f'Observado Z = {z_stat:.3f}')
    plt.title(title, fontsize=15); plt.legend(fontsize=10); plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(os.path.join(GRAFICOS_OUTPUT_DIR, f"{filename_base}_z_dist.png"), dpi=300, bbox_inches='tight')
    plt.close()

def display_f_distribution(f_stat, f_lower_crit, f_upper_crit, dfn, dfd, title, filename_base):
    if dfn <= 0 or dfd <= 0: return
    x_max_percentile = 0.999
    if f_stat > po.stats.f.ppf(x_max_percentile, dfn, dfd) or f_upper_crit > po.stats.f.ppf(x_max_percentile, dfn, dfd):
        x_max_percentile = 0.99999
        if f_stat > po.stats.f.ppf(x_max_percentile, dfn, dfd) or f_upper_crit > po.stats.f.ppf(x_max_percentile, dfn, dfd):
             x_max_percentile = 1.0 - 1e-7
    x_max_candidate = po.stats.f.ppf(x_max_percentile, dfn, dfd)
    plot_max_x = max(x_max_candidate, f_stat * 1.3, f_upper_crit * 1.3, 3.0)
    x = np.linspace(0, plot_max_x, 500); y = po.stats.f.pdf(x, dfn, dfd)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f'Distribución F (dfn={dfn}, dfd={dfd})', color=colores["azul_principal"])
    plt.xlabel("Valor F", fontsize=12); plt.ylabel("Densidad", fontsize=12)
    plt.fill_between(x, 0, y, where=(x < f_lower_crit), color=colores["naranja"], alpha=0.7, label=f'Rechazo\nF < {f_lower_crit:.3f}')
    plt.fill_between(x, 0, y, where=(x > f_upper_crit), color=colores["rosado"], alpha=0.7, label=f'Rechazo\nF > {f_upper_crit:.3f}')
    plt.axvline(f_lower_crit, color=colores["naranja_oscuro"], linestyle='--', label=f'Crítico Inf. F = {f_lower_crit:.3f}')
    plt.axvline(f_upper_crit, color=colores["rosado_oscuro"], linestyle='--', label=f'Crítico Sup. F = {f_upper_crit:.3f}')
    plt.axvline(f_stat, color=colores["azul_observado"], linestyle='-', lw=2.5, label=f'Observado F = {f_stat:.3f}')
    plt.title(title, fontsize=15); plt.legend(fontsize=10); plt.grid(True, linestyle=':', alpha=0.7)
    plt.xlim(left=0, right=plot_max_x); plt.ylim(bottom=0)
    plt.savefig(os.path.join(GRAFICOS_OUTPUT_DIR, f"{filename_base}_f_dist.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print(f"--- INICIO: GRÁFICOS SE GUARDARÁN EN '{GRAFICOS_OUTPUT_DIR}' ---")
    
    print("\n--- GRÁFICOS DESCRIPTIVOS ---")
    hist_nivel_educativo(data_2020, EDU_2020_NIVEL_COL, 2020)
    hist_nivel_educativo(data_2022, EDU_2022_NIVEL_COL, 2022)
    barras_ingreso_promedio(data_2020, data_2022, ING_COL)
    regresion_y_grafico(data_2020, HRS_COL, ING_COL, "2020")
    regresion_y_grafico(data_2022, HRS_COL, ING_COL, "2022")
    regresion_y_grafico(data_2020, EDU_2020_ANOS_COL, ING_COL, "2020 (Años Estudio)")
    regresion_y_grafico(data_2022, EDU_2022_ANOS_COL, ING_COL, "2022 (Años Estudio)")

    print("\n--- PRUEBAS Y GRÁFICOS DE DISTRIBUCIÓN ---")
    
    umbral_edu_p1 = T_CRITICO_val
    # 2020
    temp_df_p1_2020 = data_2020.copy()
    temp_df_p1_2020[ING_COL] = pd.to_numeric(temp_df_p1_2020[ING_COL], errors='coerce')
    temp_df_p1_2020[EDU_COL_P1_2020] = pd.to_numeric(temp_df_p1_2020[EDU_COL_P1_2020], errors='coerce')
    mayores_p1_2020 = temp_df_p1_2020[temp_df_p1_2020[EDU_COL_P1_2020] > umbral_edu_p1][ING_COL].dropna()
    menores_p1_2020 = temp_df_p1_2020[temp_df_p1_2020[EDU_COL_P1_2020] <= umbral_edu_p1][ING_COL].dropna()
    if len(mayores_p1_2020) >= 2 and len(menores_p1_2020) >= 2:
        resultado_p1_2020 = po.comparar_ingresos_por_x(data_2020, EDU_COL_P1_2020, umbral_edu_p1, 2020)
        if resultado_p1_2020 and 't' in resultado_p1_2020:
            df = max(1, min(len(mayores_p1_2020) - 1, len(menores_p1_2020) - 1))
            titulo = f"P1 {obtener_nombre_descriptivo(ING_COL)}: {obtener_nombre_descriptivo(EDU_COL_P1_2020, 2020)} > {umbral_edu_p1:.2f} vs. Resto (2020)"
            fname_base = f"P1_2020_Ingreso_vs_Educ_{EDU_COL_P1_2020}_gt{str(umbral_edu_p1).replace('.', 'p')}"
            display_t_distribution(resultado_p1_2020['t'], T_CRITICO_val, df, titulo, fname_base, tail_type='right')
    # 2022
    temp_df_p1_2022 = data_2022.copy()
    temp_df_p1_2022[ING_COL] = pd.to_numeric(temp_df_p1_2022[ING_COL], errors='coerce')
    temp_df_p1_2022[EDU_COL_P1_2022] = pd.to_numeric(temp_df_p1_2022[EDU_COL_P1_2022], errors='coerce')
    mayores_p1_2022 = temp_df_p1_2022[temp_df_p1_2022[EDU_COL_P1_2022] > umbral_edu_p1][ING_COL].dropna()
    menores_p1_2022 = temp_df_p1_2022[temp_df_p1_2022[EDU_COL_P1_2022] <= umbral_edu_p1][ING_COL].dropna()
    if len(mayores_p1_2022) >= 2 and len(menores_p1_2022) >= 2:
        resultado_p1_2022 = po.comparar_ingresos_por_x(data_2022, EDU_COL_P1_2022, umbral_edu_p1, 2022)
        if resultado_p1_2022 and 't' in resultado_p1_2022:
            df = max(1, min(len(mayores_p1_2022) - 1, len(menores_p1_2022) - 1))
            titulo = f"P1 {obtener_nombre_descriptivo(ING_COL)}: {obtener_nombre_descriptivo(EDU_COL_P1_2022, 2022)} > {umbral_edu_p1:.2f} vs. Resto (2022)"
            fname_base = f"P1_2022_Ingreso_vs_Educ_{EDU_COL_P1_2022}_gt{str(umbral_edu_p1).replace('.', 'p')}"
            display_t_distribution(resultado_p1_2022['t'], T_CRITICO_val, df, titulo, fname_base, tail_type='right')

    for año_num, df_orig, edu_col_niv in [(2020, data_2020, EDU_2020_NIVEL_COL), (2022, data_2022, EDU_2022_NIVEL_COL)]:
        print(f"\nP1B {obtener_nombre_descriptivo(ING_COL)} entre Pares de {obtener_nombre_descriptivo(edu_col_niv, año_num)} ({año_num})")
        resultados_p1b_list = po.comparar_pares_niveles(df_orig, edu_col_niv, año_num)
        idx_res = 0
        for a, b in pairs_list:
            temp_df_p1b = df_orig.copy()
            temp_df_p1b[ING_COL] = pd.to_numeric(temp_df_p1b[ING_COL], errors='coerce')
            temp_df_p1b[edu_col_niv] = pd.to_numeric(temp_df_p1b[edu_col_niv], errors='coerce')
            grupo1 = temp_df_p1b[temp_df_p1b[edu_col_niv] == a][ING_COL].dropna()
            grupo2 = temp_df_p1b[temp_df_p1b[edu_col_niv] == b][ING_COL].dropna()
            if len(grupo1) >= 2 and len(grupo2) >= 2 and idx_res < len(resultados_p1b_list):
                res_par = resultados_p1b_list[idx_res]
                if 't' in res_par and 'f' in res_par:
                    df_t_p1b = max(1, min(len(grupo1) - 1, len(grupo2) - 1))
                    titulo_t = f"P1B {año_num}: {level_map_dict.get(a,a)} vs {level_map_dict.get(b,b)} (Ingreso T)"
                    fname_t = f"P1B_{año_num}_{a}vs{b}_Ingreso"
                    display_t_distribution(res_par['t'], T_CRITICO_val, df_t_p1b, titulo_t, fname_t, tail_type='two')
                    dfn_p1b, dfd_p1b = len(grupo2) - 1, len(grupo1) - 1
                    if dfn_p1b > 0 and dfd_p1b > 0:
                        titulo_f = f"P1B {año_num}: {level_map_dict.get(a,a)} vs {level_map_dict.get(b,b)} (Varianza Ing. F)"
                        fname_f = f"P1B_{año_num}_{a}vs{b}_VarianzaIng"
                        display_f_distribution(res_par['f'], F_LOWER_val, F_UPPER_val, dfn_p1b, dfd_p1b, titulo_f, fname_f)
                idx_res +=1
            
    for año_num, df_orig, edu_col_niv, sexo_c in [(2020, data_2020, EDU_2020_NIVEL_COL, SEX_COL_2020), \
                                               (2022, data_2022, EDU_2022_NIVEL_COL, SEX_COL_2022)]:
        for sexo_nombre_loop, sexo_valor_map in [("Mujeres", 2), ("Hombres", 1)]:
            print(f"\nP3 Proporción de {sexo_nombre_loop} entre Pares de {obtener_nombre_descriptivo(edu_col_niv, año_num)} ({año_num})")
            resultados_p3_list = po.comparar_proporciones_sexo(df_orig, edu_col_niv, sexo_c, año_num, sexo_nombre_loop.lower(), sexo_valor_map)
            idx_res_p3 = 0
            for a, b in pairs_list:
                if idx_res_p3 < len(resultados_p3_list):
                    res_par_p3 = resultados_p3_list[idx_res_p3]
                    if 'z' in res_par_p3:
                        titulo_z = f"P3 {año_num} {sexo_nombre_loop}: {level_map_dict.get(a,a)} vs {level_map_dict.get(b,b)} (Proporción Z)"
                        fname_z = f"P3_{año_num}_{sexo_nombre_loop}_{a}vs{b}_Prop"
                        display_z_distribution(res_par_p3['z'], Z_CRITICO_val, titulo_z, fname_z, tail_type='left')
                    idx_res_p3 += 1
                else: break

    umbral_edu_p4 = T_CRITICO_val
    resultado_p4 = po.comparar_proporcion_general(data_2020, data_2022, EDU_2020_ANOS_COL, EDU_2022_ANOS_COL, umbral_edu_p4)
    if resultado_p4 and 'z' in resultado_p4:
        titulo_p4 = f"P4 Prop. Personas con >{umbral_edu_p4:.2f} {obtener_nombre_descriptivo(EDU_2020_ANOS_COL).split('(')[0].strip()} (General 2020 vs 2022)"
        fname_p4 = f"P4_Prop_EducAnos_gt{str(umbral_edu_p4).replace('.', 'p')}_General"
        display_z_distribution(resultado_p4['z'], Z_CRITICO_val, titulo_p4, fname_p4, tail_type='left')

    for grupo_cfg in [{"grupo": None, "sexo1_col_cfg": None, "sexo2_col_cfg": None, "nombre_graf": "Ambos Sexos"},
                      {"grupo": "mujeres", "sexo1_col_cfg": SEX_COL_2020, "sexo2_col_cfg": SEX_COL_2022, "nombre_graf": "Mujeres"},
                      {"grupo": "hombres", "sexo1_col_cfg": SEX_COL_2020, "sexo2_col_cfg": SEX_COL_2022, "nombre_graf": "Hombres"}]:
        d1_temp_p5 = data_2020.copy(); d2_temp_p5 = data_2022.copy()
        if grupo_cfg["grupo"]:
            sexo_val = 2 if grupo_cfg["grupo"] == "mujeres" else 1
            d1_temp_p5 = d1_temp_p5[d1_temp_p5[grupo_cfg["sexo1_col_cfg"]] == sexo_val]
            d2_temp_p5 = d2_temp_p5[d2_temp_p5[grupo_cfg["sexo2_col_cfg"]] == sexo_val]
        x1_edu_p5 = pd.to_numeric(d1_temp_p5[EDU_2020_ANOS_COL], errors='coerce').dropna()
        x2_edu_p5 = pd.to_numeric(d2_temp_p5[EDU_2022_ANOS_COL], errors='coerce').dropna()
        if len(x1_edu_p5) >= 2 and len(x2_edu_p5) >= 2:
            resultado_p5 = po.comparar_media_educacion(data_2020, data_2022, EDU_2020_ANOS_COL, EDU_2022_ANOS_COL,
                                                     grupo=grupo_cfg["grupo"], sexo1=grupo_cfg["sexo1_col_cfg"], sexo2=grupo_cfg["sexo2_col_cfg"])
            if resultado_p5 and 't' in resultado_p5:
                df_p5 = max(1, min(len(x1_edu_p5) - 1, len(x2_edu_p5) - 1))
                titulo_p5 = f"P5 Media de {obtener_nombre_descriptivo(EDU_2020_ANOS_COL).split('(')[0].strip()} ({grupo_cfg['nombre_graf']}): 2020 vs 2022"
                fname_p5 = f"P5_Media_EducAnos_{grupo_cfg['nombre_graf'].replace(' ','')}"
                display_t_distribution(resultado_p5['t'], T_CRITICO_val, df_p5, titulo_p5, fname_p5, tail_type='right')

    h1_p6 = pd.to_numeric(data_2020[HRS_COL], errors='coerce').dropna()
    h2_p6 = pd.to_numeric(data_2022[HRS_COL], errors='coerce').dropna()
    if len(h1_p6) >= 2 and len(h2_p6) >= 2:
        resultado_p6 = po.comparar_horas_trabajadas(data_2020, data_2022, HRS_COL)
        if resultado_p6 and 't' in resultado_p6:
            df_p6 = max(1, min(len(h1_p6) - 1, len(h2_p6) - 1))
            titulo_p6 = f"P6 Media de {obtener_nombre_descriptivo(HRS_COL)}: 2020 vs 2022"
            fname_p6 = "P6_Media_HorasTrabajadas"
            display_t_distribution(resultado_p6['t'], T_CRITICO_val, df_p6, titulo_p6, fname_p6, tail_type='right')

    print(f"\n--- FIN: TODOS LOS GRÁFICOS GUARDADOS EN '{GRAFICOS_OUTPUT_DIR}' ---")