import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import os

data_2020_raw = pd.read_csv("c:/Users/catic/OneDrive/Escritorio/cata/ANDES/20251/ana_stats/datos/2020.csv", sep=",", decimal=",", na_values=["NA"], quotechar='"', encoding='utf-8', low_memory=False)
data_2020_raw.columns = data_2020_raw.columns.str.lower()

data_2022_raw = pd.read_csv("c:/Users/catic/OneDrive/Escritorio/cata/ANDES/20251/ana_stats/datos/2022.csv", sep=';', decimal=",", na_values=["NA", " "], encoding='utf-8', low_memory=False)
data_2022_raw.columns = data_2022_raw.columns.str.lower()

data_2020 = data_2020_raw.copy()
data_2022 = data_2022_raw.copy()

ING_COL = 'ingtot'
EDU_COL_2020_YRS = 'p3042s1' # Años de estudio
EDU_COL_2022_YRS = 'p6210s1' # Años de estudio
EDAD_COL = 'p6040'       # Edad es la misma para ambos
SEX_COL = 'p6020'        # Sexo es la misma para ambos
HRS_COL = 'p6800'        # Horas es la misma para ambos


GRAFICOS_DIR = "Analisis_General_Graficos"
if not os.path.exists(GRAFICOS_DIR):
    os.makedirs(GRAFICOS_DIR)

def preprocesar_df(df, año):
    df_proc = df.copy() # Trabajar sobre una copia dentro de la función
    
    # Ingreso
    if ING_COL in df_proc.columns:
        df_proc[ING_COL] = pd.to_numeric(df_proc[ING_COL], errors='coerce')
    
    # Educación (años)
    edu_col_yrs = EDU_COL_2020_YRS if año == 2020 else EDU_COL_2022_YRS
    if edu_col_yrs in df_proc.columns:
        df_proc[edu_col_yrs] = pd.to_numeric(df_proc[edu_col_yrs], errors='coerce')
    
    # Edad
    if EDAD_COL in df_proc.columns:
        df_proc[EDAD_COL] = pd.to_numeric(df_proc[EDAD_COL], errors='coerce')
    
    # Sexo
    if SEX_COL in df_proc.columns:
        df_proc[SEX_COL] = pd.to_numeric(df_proc[SEX_COL], errors='coerce')
    
    # Horas
    if HRS_COL in df_proc.columns:
        df_proc[HRS_COL] = pd.to_numeric(df_proc[HRS_COL], errors='coerce')
    
    # Filtrar 99 en la columna de NIVEL EDUCATIVO 'p3042' (no 'p3042s1') para 2020 si existe
    if año == 2020 and 'p3042' in df_proc.columns:
        df_proc['p3042'] = pd.to_numeric(df_proc['p3042'], errors='coerce')
        df_proc = df_proc[df_proc['p3042'] != 99].copy()
    
    return df_proc

data_2020 = preprocesar_df(data_2020, 2020)
data_2022 = preprocesar_df(data_2022, 2022)


def estadisticas_descriptivas_var(serie_numerica, nombre_variable, formato_miles=False):
    if serie_numerica is None or serie_numerica.empty:
        print(f"\n--- Estadísticas {nombre_variable} ---")
        print("No hay datos disponibles o la columna no existe.")
        return None
    print(f"\n--- Estadísticas {nombre_variable} ---")
    mean_val = serie_numerica.mean()
    median_val = serie_numerica.median()
    mode_val = serie_numerica.mode()
    mode_str = f"{mode_val.values[0]:,.2f}" if formato_miles and not mode_val.empty else (str(mode_val.values[0]) if not mode_val.empty else "N/A")
    
    print(f"Media:      {mean_val:,.2f}" if formato_miles else f"Media:      {mean_val:.2f}")
    print(f"Mediana:    {median_val:,.2f}" if formato_miles else f"Mediana:    {median_val:.2f}")
    print(f"Moda:       {mode_str}")
    print(f"Varianza:   {serie_numerica.var():,.2f}" if formato_miles else f"Varianza:   {serie_numerica.var():.2f}")
    print(f"Desv.Est.:  {serie_numerica.std():,.2f}" if formato_miles else f"Desv.Est.:  {serie_numerica.std():.2f}")
    print(f"Mínimo:     {serie_numerica.min():,.2f}" if formato_miles else f"Mínimo:     {serie_numerica.min()}")
    print(f"Máximo:     {serie_numerica.max():,.2f}" if formato_miles else f"Máximo:     {serie_numerica.max()}")
    print(f"Asimetría:  {serie_numerica.skew():.2f}")
    print(f"Curtosis:   {serie_numerica.kurtosis():.2f}")
    print(f"N datos:    {len(serie_numerica)}")
    return serie_numerica

def graficar_histograma(serie_numerica, titulo_graf, xlabel_graf, color_hist, filename, xlim_percentil=None, bins_hist=50):
    if serie_numerica is None or serie_numerica.empty: return
    plt.figure(figsize=(8,5))
    if isinstance(bins_hist, range):
         plt.hist(serie_numerica, bins=bins_hist, color=color_hist, edgecolor='black', align='left')
    else:
        plt.hist(serie_numerica, bins=bins_hist, color=color_hist, edgecolor='black')
    plt.title(titulo_graf, fontsize=14)
    plt.xlabel(xlabel_graf, fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    if xlim_percentil:
        min_val = serie_numerica.min()
        plt.xlim(min_val if min_val < 0 else 0, np.percentile(serie_numerica, xlim_percentil))
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.savefig(os.path.join(GRAFICOS_DIR, filename), dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()

def graficar_boxplot(serie_numerica, titulo_graf, filename, ylabel_graf="Valor"):
    if serie_numerica is None or serie_numerica.empty: return
    plt.figure(figsize=(6,7))
    sns.boxplot(y=serie_numerica, color='lightblue')
    plt.title(titulo_graf, fontsize=14)
    plt.ylabel(ylabel_graf, fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.savefig(os.path.join(GRAFICOS_DIR, filename), dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()

def estadisticas_sexo(df, sexo_col_name, año):
    if sexo_col_name not in df.columns:
        print(f"\n--- Estadísticas Sexo --- \nColumna '{sexo_col_name}' no encontrada para el año {año}.")
        return None
    sexo = df[sexo_col_name].dropna()
    if sexo.empty:
        print(f"\n--- Estadísticas Sexo ({sexo_col_name}) - Año {año} --- \nNo hay datos.")
        return None
        
    counts = sexo.value_counts().sort_index()
    print(f"\n--- Estadísticas Sexo ({sexo_col_name}) - Año {año} ---")
    print(counts)
    
    labels = counts.index.map(lambda x: 'Hombre' if x==1.0 else ('Mujer' if x==2.0 else f'Cod_{x}')).tolist()

    plt.figure(figsize=(6,5))
    counts.plot(kind='bar', color=['cornflowerblue', 'lightcoral'], edgecolor='black')
    plt.title(f'Distribución por Sexo - Año {año}', fontsize=14)
    plt.xlabel('Sexo', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=0)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    filename = f"distribucion_sexo_{año}.png"
    plt.savefig(os.path.join(GRAFICOS_DIR, filename), dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()
    return sexo

def analisis_correlaciones(df, columnas_numericas, año):
    columnas_existentes = [col for col in columnas_numericas if col in df.columns]
    if len(columnas_existentes) < 2:
        print(f"No hay suficientes columnas numéricas existentes para calcular correlaciones en el año {año}.")
        return

    df_numericas = df[columnas_existentes].copy()
    for col in df_numericas.columns: # Asegurar que todas sean numéricas para .corr()
        df_numericas[col] = pd.to_numeric(df_numericas[col], errors='coerce')
    df_numericas.dropna(inplace=True)

    if df_numericas.empty or len(df_numericas.columns) < 2:
        print(f"No hay suficientes datos/columnas para la matriz de correlación del {año} después de limpiar NaNs.")
        return

    corr_matrix = df_numericas.corr()
    print(f"\n--- Matriz de Correlación - Año {año} ---")
    print(corr_matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
    plt.title(f'Matriz de Correlación - Año {año}', fontsize=15)
    filename = f"matriz_correlacion_{año}.png"
    plt.savefig(os.path.join(GRAFICOS_DIR, filename), dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()

def graficar_dispersion(df, x_col, y_col, año, alpha=0.2, sample_n=None):
    if not (x_col in df.columns and y_col in df.columns):
        print(f"Columnas {x_col} o {y_col} no encontradas para el gráfico de dispersión del año {año}.")
        return

    df_plot = df[[x_col, y_col]].copy()
    df_plot[x_col] = pd.to_numeric(df_plot[x_col], errors='coerce')
    df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors='coerce')
    df_plot.dropna(inplace=True)

    if df_plot.empty:
        print(f"No hay datos para el gráfico de dispersión {x_col} vs {y_col} del año {año} después de limpiar NaNs.")
        return

    if sample_n and len(df_plot) > sample_n:
        df_plot = df_plot.sample(n=sample_n, random_state=1)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_col, y=y_col, data=df_plot, alpha=alpha, edgecolor=None, s=30)
    correlation = df_plot[x_col].corr(df_plot[y_col]) if len(df_plot[x_col].unique()) > 1 and len(df_plot[y_col].unique()) > 1 else float('nan')
    plt.title(f'Dispersión: {y_col} vs {x_col} - Año {año} (Corr: {correlation:.2f})', fontsize=14)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    filename = f"dispersion_{y_col}_vs_{x_col}_{año}.png"
    plt.savefig(os.path.join(GRAFICOS_DIR, filename), dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()

def boxplot_por_categoria(df, num_col, cat_col, año, cat_map=None):
    if not (num_col in df.columns and cat_col in df.columns):
        print(f"Columnas {num_col} o {cat_col} no encontradas para el boxplot del año {año}.")
        return
        
    df_plot = df[[num_col, cat_col]].copy()
    df_plot[num_col] = pd.to_numeric(df_plot[num_col], errors='coerce')
    df_plot[cat_col] = pd.to_numeric(df_plot[cat_col], errors='coerce', downcast='integer') # Tratar como entero si es posible
    df_plot.dropna(inplace=True)

    if df_plot.empty:
        print(f"No hay datos para el boxplot {num_col} por {cat_col} del año {año} después de limpiar NaNs.")
        return

    cat_col_mapped_name = cat_col + "_nombre" # Para no sobreescribir la original
    if cat_map:
        df_plot[cat_col_mapped_name] = df_plot[cat_col].map(cat_map).fillna("Otro")
    else:
        df_plot[cat_col_mapped_name] = df_plot[cat_col].astype(str) # Usar como string si no hay mapa

    plt.figure(figsize=(10, 7))
    sns.boxplot(x=cat_col_mapped_name, y=num_col, data=df_plot, palette="pastel")
    plt.title(f'Distribución de {num_col} por {cat_col} - Año {año}', fontsize=14)
    plt.xlabel(cat_col.replace('_',' ').title(), fontsize=12)
    plt.ylabel(num_col.replace('_',' ').title(), fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    filename = f"boxplot_{num_col}_por_{cat_col}_{año}.png"
    plt.savefig(os.path.join(GRAFICOS_DIR, filename), dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()

def analisis_completo_variable_numerica(df, col_name, año, es_ingreso=False):
    print(f"\n===== ANÁLISIS DETALLADO: {col_name.upper()} - AÑO {año} =====")
    if col_name not in df.columns:
        print(f"Columna '{col_name}' no encontrada en los datos del año {año}.")
        return

    serie = df[col_name].dropna()
    if serie.empty:
        print("No hay datos para analizar.")
        return

    estadisticas_descriptivas_var(serie, f"{col_name} ({año})", formato_miles=es_ingreso)
    
    bins_hist = 100 if es_ingreso else (range(int(serie.min()), int(serie.max()) + 2) if serie.nunique() < 30 and serie.max() < 150 else 50)
    xlim_perc = 99.5 if es_ingreso else None # Ajustado para ingreso
    graficar_histograma(serie, f'Histograma {col_name} ({año})', col_name, 'cornflowerblue', 
                        f"hist_{col_name}_{año}.png", xlim_percentil=xlim_perc, bins_hist=bins_hist)
    
    graficar_boxplot(serie, f'Boxplot {col_name} ({año})', f"boxplot_{col_name}_{año}.png", col_name)
    
    if len(serie) > 20: # qqplot y shapiro necesitan suficientes datos
        try:
            fig_qq = sm.qqplot(serie, line='s')
            plt.title(f'QQ-Plot de {col_name} ({año}) vs Normal', fontsize=14)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.savefig(os.path.join(GRAFICOS_DIR, f"qqplot_{col_name}_{año}.png"), dpi=300, bbox_inches='tight')
            #plt.show()
            plt.close(fig_qq)
        except Exception as e:
            print(f"No se pudo generar QQ-Plot para {col_name} ({año}): {e}")

        if len(serie) <= 5000: # Shapiro-Wilk
            stat_shapiro, p_shapiro = stats.shapiro(serie)
            print(f"Prueba de Normalidad Shapiro-Wilk para {col_name}:")
            print(f"  Estadístico={stat_shapiro:.3f}, p-valor={p_shapiro:.3g} -> {'Probablemente NO Normal' if p_shapiro < 0.05 else 'No se puede rechazar Normalidad'}")
        else: # D'Agostino's K^2 para muestras más grandes
            stat_dagostino, p_dagostino = stats.normaltest(serie)
            print(f"Prueba de Normalidad D'Agostino's K^2 para {col_name}:")
            print(f"  Estadístico={stat_dagostino:.3f}, p-valor={p_dagostino:.3g} -> {'Probablemente NO Normal' if p_dagostino < 0.05 else 'No se puede rechazar Normalidad'}")


# --- Función Principal de Análisis ---
def resumen_general_ampliado(df, año):
    print(f"\n\n========== ANÁLISIS GENERAL - AÑO {año} ==========")
    
    edu_col_actual_yrs = EDU_COL_2020_YRS if año == 2020 else EDU_COL_2022_YRS
    
    print("\n--- Resumen de Valores Faltantes (primeras 20 columnas) ---")
    print(df.isnull().sum().head(20))

    analisis_completo_variable_numerica(df, ING_COL, año, es_ingreso=True)
    analisis_completo_variable_numerica(df, edu_col_actual_yrs, año)
    analisis_completo_variable_numerica(df, EDAD_COL, año)
    analisis_completo_variable_numerica(df, HRS_COL, año)

    estadisticas_sexo(df, SEX_COL, año)

    cols_numericas_corr = [ING_COL, edu_col_actual_yrs, EDAD_COL, HRS_COL]
    # Filtrar solo las columnas que realmente existen en el df para evitar errores
    cols_numericas_corr_existentes = [col for col in cols_numericas_corr if col in df.columns]
    if len(cols_numericas_corr_existentes) >=2:
        analisis_correlaciones(df, cols_numericas_corr_existentes, año)

    sample_size_scatter = 5000 
    if edu_col_actual_yrs in df.columns and ING_COL in df.columns:
        graficar_dispersion(df, edu_col_actual_yrs, ING_COL, año, sample_n=sample_size_scatter)
    if EDAD_COL in df.columns and ING_COL in df.columns:
        graficar_dispersion(df, EDAD_COL, ING_COL, año, sample_n=sample_size_scatter)
    if HRS_COL in df.columns and ING_COL in df.columns:
        graficar_dispersion(df, HRS_COL, ING_COL, año, sample_n=sample_size_scatter)
    
    if SEX_COL in df.columns and ING_COL in df.columns:
        sex_map = {1.0: 'Hombre', 2.0: 'Mujer'}
        boxplot_por_categoria(df, ING_COL, SEX_COL, año, cat_map=sex_map)

#RUN
if __name__ == "__main__":
    resumen_general_ampliado(data_2020, 2020)
    resumen_general_ampliado(data_2022, 2022)
