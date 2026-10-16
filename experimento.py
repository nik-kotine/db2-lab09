import time
import importlib.util

import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import SnowballStemmer

import connect

spec = importlib.util.spec_from_file_location("inverted_index", "inverted-index.py")
inverted_index = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inverted_index)

InvertedIndex = inverted_index.InvertedIndex
AND = inverted_index.AND
OR = inverted_index.OR
AND_NOT = inverted_index.AND_NOT

stemmer = SnowballStemmer("spanish")

consultas_booleanas = [
    ("economía", "AND", "inflación"),
    ("banca", "OR", "finanzas"),
    ("tecnología", "AND-NOT", "inflación"),
]

consultas_coseno = [
    "innovaciones en banca digital y tecnología financiera",
    "inflación y crecimiento económico",
    "sostenibilidad y energías renovables",
]

REPETICIONES_BOOLEANA = 1000
REPETICIONES_COSENO = 10
SEMILLA = 42


def cargar_bow():

    conn = connect.connect_db()
    df = pd.read_sql("SELECT id, bag_of_words FROM noticias ORDER BY id;", conn)
    conn.close()
    return df


def construir_indice(df):
    idx = InvertedIndex()
    for _, fila in df.iterrows():
        doc_id = f"doc{fila['id']}"
        bow = fila["bag_of_words"]
        idx._docs[doc_id] = bow
        for palabra, freq in bow.items():
            if palabra not in idx.index:
                idx.index[palabra] = []
            idx.index[palabra].append((doc_id, freq))
            idx.idf[palabra] = idx.idf.get(palabra, 0) + 1
    return idx


def booleana_lineal(idx, palabra1, operador, palabra2):
    p1 = stemmer.stem(palabra1)
    p2 = stemmer.stem(palabra2)
    resultado = []
    for doc_id, bow in idx._docs.items():
        esta1 = p1 in bow
        esta2 = p2 in bow
        if operador == "AND":
            cumple = esta1 and esta2
        elif operador == "OR":
            cumple = esta1 or esta2
        else:
            cumple = esta1 and not esta2
        if cumple:
            resultado.append(doc_id)
    return resultado


def booleana_indice(idx, palabra1, operador, palabra2):
    lista1 = idx.L(palabra1)
    lista2 = idx.L(palabra2)
    if operador == "AND":
        resultado = AND(lista1, lista2)
    elif operador == "OR":
        resultado = OR(lista1, lista2)
    else:
        resultado = AND_NOT(lista1, lista2)
    return [doc_id for doc_id, freq in resultado]


def tiempo_booleana(idx, usar_indice):
    total = 0
    for consulta in consultas_booleanas:
        for _ in range(REPETICIONES_BOOLEANA):
            inicio = time.perf_counter()
            if usar_indice:
                booleana_indice(idx, *consulta)
            else:
                booleana_lineal(idx, *consulta)
            total += time.perf_counter() - inicio
    return total / (len(consultas_booleanas) * REPETICIONES_BOOLEANA)


def tiempo_coseno(idx, usar_indice):
    total = 0
    for consulta in consultas_coseno:
        for _ in range(REPETICIONES_COSENO):
            inicio = time.perf_counter()
            if usar_indice:
                idx.cosine_search(consulta, 5)
            else:
                idx.cosine_search_linear(consulta, 5)
            total += time.perf_counter() - inicio
    return total / (len(consultas_coseno) * REPETICIONES_COSENO)


def verificar(idx):
    print("\nVerificación de resultados (colección completa)")
    for consulta in consultas_booleanas:
        a = set(booleana_lineal(idx, *consulta))
        b = set(booleana_indice(idx, *consulta))
        print(consulta, "-> iguales:", a == b, "| documentos:", len(a))
    for consulta in consultas_coseno:
        lineal = idx.cosine_search_linear(consulta, 5)
        indice = idx.cosine_search(consulta, 5)
        diferencia = max(abs(x[1] - y[1]) for x, y in zip(lineal, indice))
        print(consulta, "-> diferencia máxima de score:", diferencia)


def graficar(tamanos, lineal, indice, titulo, archivo):
    plt.figure(figsize=(13, 6))
    plt.plot(tamanos, lineal, marker="o", label="Lineal")
    plt.plot(tamanos, indice, marker="o", label="Índice invertido")
    for x, y in zip(tamanos, lineal):
        plt.annotate(f"{y:.3g}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")
    for x, y in zip(tamanos, indice):
        plt.annotate(f"{y:.3g}", (x, y), textcoords="offset points", xytext=(0, -15), ha="center")
    plt.margins(y=0.15)
    plt.title(titulo)
    plt.xlabel("Número de documentos")
    plt.ylabel("Tiempo promedio (segundos)")
    plt.legend()
    plt.grid(True)
    plt.savefig(archivo, dpi=150)
    plt.show()


if __name__ == "__main__":
    df = cargar_bow()
    df = df.sample(frac=1, random_state=SEMILLA).reset_index(drop=True)

    total = len(df)
    tamanos = [10, 25, 50, 100, 250, 500, 750, 1000, total]
    tamanos = [t for t in tamanos if t <= total]

    resultados = []
    for n in tamanos:
        subconjunto = df.head(n).sort_values("id")
        idx = construir_indice(subconjunto)

        bool_lineal = tiempo_booleana(idx, False)
        bool_indice = tiempo_booleana(idx, True)
        cos_lineal = tiempo_coseno(idx, False)
        cos_indice = tiempo_coseno(idx, True)

        resultados.append({
            "n_documentos": n,
            "booleana_lineal": bool_lineal,
            "booleana_indice": bool_indice,
            "coseno_lineal": cos_lineal,
            "coseno_indice": cos_indice,
        })
        print(f"n={n} listo")

    tabla = pd.DataFrame(resultados)
    print(tabla.to_string(index=False))
    tabla.to_csv("resultados_experimento.csv", index=False)

    verificar(idx)

    graficar(
        tabla["n_documentos"],
        tabla["booleana_lineal"],
        tabla["booleana_indice"],
        "Búsqueda booleana: lineal vs índice invertido",
        "grafico_booleana.png",
    )
    graficar(
        tabla["n_documentos"],
        tabla["coseno_lineal"],
        tabla["coseno_indice"],
        "Similitud de coseno: lineal vs índice invertido",
        "grafico_coseno.png",
    )
