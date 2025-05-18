import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math as math 
from scipy import stats 
from statsmodels.stats.proportion import proportions_ztest


data_2020 = pd.read_csv("c:/Users/catic/OneDrive/Escritorio/cata/ANDES/20251/ana_stats/datos/2020.csv", sep=",", decimal=",", na_values=["NA"], quotechar='"', encoding='utf-8', low_memory=False)
data_2022 = pd.read_csv("c:/Users/catic/OneDrive/Escritorio/cata/ANDES/20251/ana_stats/datos/2022.csv", sep=';', decimal=",", na_values=["NA", " "], encoding='utf-8', low_memory=False)



# Variables clave
level_map = {
    0: "Ninguno",
    1: "Preescolar",
    2: "Primaria",
    3: "Secundaria",
    4: "Media",
    5: "Superior",
}

pairs = [(0,1),(1,2),(2,4),(4,5)] 

EDU_2020 = 'p3042'
EDU_2022 = 'p6210'
ING = 'ingtot'
SEX_2020 = 'p3271'  # mujeres = 2
SEX_2022 = 'p6020'
HRS = 'p6800'

#valores de referencia q ana quiere 
F_LOWER = 0.96155504
F_UPPER = 1.03998207
Z_CRITICO = 1.645
T_CRITICO = 1.6449

#conversiones 
data_2020[ING] = pd.to_numeric(data_2020[ING], errors='coerce')
data_2022[ING] = pd.to_numeric(data_2022[ING], errors='coerce')
data_2020 = data_2020.dropna(subset=[ING])
data_2022 = data_2022.dropna(subset=[ING])
data_2020 = data_2020[data_2020[EDU_2020] != 99]

#PREGUNTA 1 T
def comparar_ingresos_por_x(df, edu_col, x, año):
    df[ING] = pd.to_numeric(df[ING], errors='coerce')
    df[edu_col] = pd.to_numeric(df[edu_col], errors='coerce')
    mayores = df[df[edu_col] > x][ING].dropna()
    menores = df[df[edu_col] <= x][ING].dropna()
    if len(mayores) >= 2 and len(menores) >= 2:
        t, p = stats.ttest_ind(mayores, menores, equal_var=False)
        print(f"\nAÑO {año} — Educación > {x} vs <= {x}")
        print(f"Media mayores: {mayores.mean():.2f} | menores: {menores.mean():.2f}")
        print("→ Se RECHAZA H₀" if t > T_CRITICO else "→ NO se rechaza H₀")
        return {'año': año, 'mayores': mayores.mean(), 'menores': menores.mean(), 't': t, 'p': p}
    else:
        print(f"AÑO {año}: datos insuficientes para comparación")
        return None

#PREGUNTA 1B T y F
def comparar_pares_niveles(df, edu_col, año):
    df[ING] = pd.to_numeric(df[ING], errors='coerce')
    df[edu_col] = pd.to_numeric(df[edu_col], errors='coerce')
    print(f"\n► Pregunta 1B — AÑO {año}")
    resultados = []
    for a, b in pairs:
        grupo1 = df[df[edu_col] == a][ING].dropna()
        grupo2 = df[df[edu_col] == b][ING].dropna()
        if len(grupo1) >= 2 and len(grupo2) >= 2:
            t, p = stats.ttest_ind(grupo1, grupo2, equal_var=False)
            f = np.var(grupo2, ddof=1) / np.var(grupo1, ddof=1)
            print(f"{level_map[a]} vs {level_map[b]}:")
            print(f"  T = {t:.2f}, p = {p:.4g} → {'RECHAZA H₀' if abs(t) > T_CRITICO else 'NO se rechaza H₀'}")
            print(f"  F = {f:.3f} → {'varianzas similares' if F_LOWER <= f <= F_UPPER else 'varianzas diferentes'}")
            resultados.append({'nivel1': level_map[a], 'nivel2': level_map[b], 't': t, 'p': p, 'f': f})
        else:
            print(f"{level_map[a]} vs {level_map[b]}: datos insuficientes")
    return resultados

#PREGUNTA 2
def varianza_por_nivel(df, edu_col):
    df[ING] = pd.to_numeric(df[ING], errors='coerce')
    df[edu_col] = pd.to_numeric(df[edu_col], errors='coerce')
    print("\nVarianzas de ingreso por nivel:")
    var_dict = {}
    for nivel in sorted(level_map.keys()):
        grupo = df[df[edu_col] == nivel][ING].dropna()
        if len(grupo) > 1:
            var = np.var(grupo, ddof=1)
            var_dict[level_map[nivel]] = var
            print(f"{level_map[nivel]}: {var:.2f} (n={len(grupo)})")
        else:
            print(f"{level_map[nivel]}: datos insuficientes")
    return var_dict

#PREGUNTA 3
def comparar_proporciones_sexo(df, edu_col, sexo_col, año, sexo_nombre, sexo_valor):
    df[edu_col] = pd.to_numeric(df[edu_col], errors='coerce')
    df[sexo_col] = pd.to_numeric(df[sexo_col], errors='coerce')
    print(f"\nAÑO {año} — Proporciones de {sexo_nombre.upper()}")
    resultados = []
    for a, b in pairs:
        grupo1 = df[df[edu_col] == a]
        grupo2 = df[df[edu_col] == b]
        count1 = (grupo1[sexo_col] == sexo_valor).sum()
        count2 = (grupo2[sexo_col] == sexo_valor).sum()
        n1, n2 = len(grupo1), len(grupo2)
        if n1 >= 2 and n2 >= 2 and count1 > 0 and count2 > 0:
            z, pval = proportions_ztest([count1, count2], [n1, n2], alternative='smaller')
            print(f"{level_map[a]} vs {level_map[b]}: Z = {z:.2f}, p = {pval:.4g} → {'RECHAZA H₀' if z < -Z_CRITICO else 'NO se rechaza H₀'}")
            resultados.append({'nivel1': level_map[a], 'nivel2': level_map[b], 'z': z, 'p': pval})
        else:
            print(f"{level_map[a]} vs {level_map[b]}: datos insuficientes")
    return resultados
            

#PREGUNTA 4 y 5 
#solo un genero
def comparar_media_educacion(df1, df2, col1, col2, grupo=None, sexo1=None, sexo2=None):
    d1 = df1.copy()
    d2 = df2.copy()
    if grupo:
        d1 = d1[d1[sexo1] == (2 if grupo == "mujeres" else 1)]
        d2 = d2[d2[sexo2] == (2 if grupo == "mujeres" else 1)]
    d1[col1] = pd.to_numeric(d1[col1], errors='coerce')
    d2[col2] = pd.to_numeric(d2[col2], errors='coerce')
    x1, x2 = d1[col1].dropna(), d2[col2].dropna()
    t, p = stats.ttest_ind(x2, x1, equal_var=False)
    nombre = grupo if grupo else "ambos"
    print(f"\n► Pregunta 5 — Media educación {nombre.upper()}")
    print(f"Media 2020: {x1.mean():.2f} | 2022: {x2.mean():.2f}")
    print(f"T = {t:.2f}, p = {p:.4g} → {'RECHAZA H₀' if t > T_CRITICO else 'NO se rechaza H₀'}")
    return {'grupo': nombre, '2020': x1.mean(), '2022': x2.mean(), 't': t, 'p': p}


#ambos generos
def comparar_proporcion_general(df1, df2, col1, col2, x):
    df1[col1] = pd.to_numeric(df1[col1], errors='coerce')
    df2[col2] = pd.to_numeric(df2[col2], errors='coerce')
    g1, g2 = df1.dropna(subset=[col1]), df2.dropna(subset=[col2])
    c1, c2 = (g1[col1] > x).sum(), (g2[col2] > x).sum()
    n1, n2 = len(g1), len(g2)
    if n1 >= 2 and n2 >= 2:
        z, p = proportions_ztest([c1, c2], [n1, n2], alternative='smaller')
        print("\n► Pregunta 4 — Proporción > x años (ambos)")
        print(f"2020: {c1}/{n1} = {c1/n1:.2%} | 2022: {c2}/{n2} = {c2/n2:.2%}")
        print(f"Z = {z:.2f}, p = {p:.4g} → {'RECHAZA H₀' if z < -Z_CRITICO else 'NO se rechaza H₀'}")
        return {'2020': c1/n1, '2022': c2/n2, 'z': z, 'p': p}
    else:
        print("Datos insuficientes para proporciones.")
        return None

#pregunta 6 T

def comparar_horas_trabajadas(df1, df2, col):
    h1 = pd.to_numeric(df1[col], errors='coerce').dropna()
    h2 = pd.to_numeric(df2[col], errors='coerce').dropna()
    t, p = stats.ttest_ind(h2, h1, equal_var=False)
    print(f"\n► Pregunta 6 — HORAS trabajadas")
    print(f"Media 2020: {h1.mean():.2f} | 2022: {h2.mean():.2f}")
    print(f"T = {t:.2f}, p = {p:.4g} → {'RECHAZA H₀' if t > T_CRITICO else 'NO se rechaza H₀'}")
    return {'2020': h1.mean(), '2022': h2.mean(), 't': t, 'p': p}
    
#RESULTADOS
if __name__ == "__main__":

    print("► Pregunta 1")
    comparar_ingresos_por_x(data_2020, 'p3042s1', T_CRITICO, 2020)
    comparar_ingresos_por_x(data_2022, 'p6210s1', T_CRITICO, 2022)

    print("\n► Pregunta 1B — Comparación entre pares de niveles")
    comparar_pares_niveles(data_2020, EDU_2020, 2020)
    comparar_pares_niveles(data_2022, EDU_2022, 2022)

    print("\n► Pregunta 2 — Variación del ingreso por nivel educativo")
    varianza_por_nivel(data_2020, EDU_2020)
    varianza_por_nivel(data_2022, EDU_2022)

    print("\n► Pregunta 3 — Mujeres")
    comparar_proporciones_sexo(data_2020, EDU_2020, SEX_2020, 2020, "mujeres", 2)
    comparar_proporciones_sexo(data_2022, EDU_2022, SEX_2022, 2022, "mujeres", 2)

    print("\n► Pregunta 3 — Hombres")
    comparar_proporciones_sexo(data_2020, EDU_2020, SEX_2020, 2020, "hombres", 1)
    comparar_proporciones_sexo(data_2022, EDU_2022, SEX_2022, 2022, "hombres", 1)

    print("\n► Pregunta 4 — Ambos géneros")
    comparar_proporcion_general(data_2020, data_2022, EDU_2020, EDU_2022, T_CRITICO)

    print("\n► Pregunta 5 — Media de educación")
    comparar_media_educacion(data_2020, data_2022, EDU_2020, EDU_2022)
    comparar_media_educacion(data_2020, data_2022, EDU_2020, EDU_2022, grupo="mujeres", sexo1=SEX_2020, sexo2=SEX_2022)
    comparar_media_educacion(data_2020, data_2022, EDU_2020, EDU_2022, grupo="hombres", sexo1=SEX_2020, sexo2=SEX_2022)


    print("\n► Pregunta 6")
    comparar_horas_trabajadas(data_2020, data_2022, HRS)


