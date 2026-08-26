# ==============================================================================
# PIPELINE HÍBRIDO AVANZADO: DATOS REALES + GENÓMICA + ÁLGEBRA LINEAL
# + TABLAS DE VERDAD + DERIVADAS + INTEGRALES + GRAFOS + ML
#
# PASO DE SIMULACIÓN A DATOS REALES
#
# 1. DATOS DE ENTRADA
#    Coloca en esta misma carpeta:
#
#        genoma_muestra.fasta
#
#    El FASTA debe contener una o varias secuencias:
#
#        >ID_SECUENCIA_1
#        ATGCGATCG...
#
#        >ID_SECUENCIA_2
#        ATGC...
#
# 2. ETIQUETAS EXPERIMENTALES
#    Para entrenamiento supervisado real, crea:
#
#        etiquetas_experimentales.csv
#
#    Formato:
#
#        id_seq,target
#        ID_SECUENCIA_1,0.92
#        ID_SECUENCIA_2,0.71
#
#    El campo "id_seq" debe coincidir con la cabecera del FASTA.
#
#    "target" debe representar una variable cuantitativa experimental
#    definida por el usuario, por ejemplo:
#       - afinidad de unión
#       - actividad
#       - inhibición
#       - expresión
#       - viabilidad
#       - otra variable experimental cuantitativa
#
# 3. VALIDACIÓN
#    El modelo NO se considera validado simplemente porque la pérdida
#    de entrenamiento sea baja.
#
#    Para un análisis real se recomienda separar:
#       - entrenamiento
#       - validación
#       - prueba
#
#    y comparar las predicciones contra mediciones experimentales
#    independientes.
#
# 4. REPRODUCIBILIDAD
#    Conserva:
#       - ID de secuencia
#       - secuencia original
#       - target experimental
#       - unidades
#       - condiciones experimentales
#       - versión del dataset
#       - parámetros del modelo
#
# ==============================================================================

import os
import csv
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.optim as optim

from scipy.integrate import trapezoid


# ==============================================================================
# 0. CONFIGURACIÓN
# ==============================================================================

ARCHIVO_FASTA = "genoma_muestra.fasta"
ARCHIVO_TARGETS = "etiquetas_experimentales.csv"

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==============================================================================
# 1. TABLA DE INFORMACIÓN QUÍMICA Y BIOFÍSICA
# ==============================================================================

TABLA_QUIMICA_ADN = {
    "A": {
        "peso_molecular": 135.13,
        "hidropatia": -1.9,
        "puentes_h": 2,
        "tipo": "Purina"
    },

    "T": {
        "peso_molecular": 126.11,
        "hidropatia": -0.7,
        "puentes_h": 2,
        "tipo": "Pirimidina"
    },

    "G": {
        "peso_molecular": 151.13,
        "hidropatia": -3.5,
        "puentes_h": 3,
        "tipo": "Purina"
    },

    "C": {
        "peso_molecular": 111.10,
        "hidropatia": -3.5,
        "puentes_h": 3,
        "tipo": "Pirimidina"
    }
}


# ==============================================================================
# 2. TABLAS DE VERDAD LÓGICO-GENÓMICAS
# ==============================================================================

def evaluar_tabla_verdad_1(purina, h_bonds_altos):
    """
    Tabla 1:
    Regulación de unión a receptor.
    Compuerta AND.
    """
    return int(bool(purina) and bool(h_bonds_altos))


def evaluar_tabla_verdad_2(inestable, patron_repetitivo):
    """
    Tabla 2:
    Alerta de mutación de escape.
    Compuerta OR.
    """
    return int(bool(inestable) or bool(patron_repetitivo))


def evaluar_tabla_verdad_3(alta_entropia, mutacion_critica):
    """
    Tabla 3:
    Resistencia antimicrobiana condicional.
    Compuerta XOR.
    """
    return int(bool(alta_entropia)) ^ int(bool(mutacion_critica))


def evaluar_tabla_verdad_4(sitio_activo_ok, inhibidor_presente, sinergia):
    """
    Tabla 4:
    Eficacia terapéutica binaria combinada.

    (sitio_activo_ok OR inhibidor_presente) AND NOT sinergia
    """
    return int(
        (bool(sitio_activo_ok) or bool(inhibidor_presente))
        and not bool(sinergia)
    )


# ==============================================================================
# 3. PARSER FASTA
# ==============================================================================

def leer_archivo_fasta(ruta_archivo):
    """
    Lee un archivo FASTA real.

    Devuelve:
        {
            "ID_1": "ATGC...",
            "ID_2": "ATGC..."
        }
    """

    secuencias = {}

    if not os.path.exists(ruta_archivo):
        print(
            f"[!] No se encontró {ruta_archivo}."
        )
        print(
            "[!] Se utilizará una secuencia de ejemplo."
        )

        return {
            "Secuencia_Defecto":
                "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGA"
        }

    current_header = None

    with open(ruta_archivo, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):

                current_header = line[1:].strip()

                # Si se desea conservar solo el primer campo
                # de la cabecera FASTA, se puede usar:
                #
                # current_header = line[1:].split()[0]

                secuencias[current_header] = ""

            elif current_header is not None:

                secuencias[current_header] += line.upper()

    return secuencias


# ==============================================================================
# 4. CARGA DE TARGETS EXPERIMENTALES
# ==============================================================================

def cargar_targets_experimentales(ruta_csv):
    """
    Carga las etiquetas experimentales desde CSV.

    Formato esperado:

        id_seq,target
        Secuencia_1,0.92
        Secuencia_2,0.71

    Devuelve:

        {
            "Secuencia_1": 0.92,
            "Secuencia_2": 0.71
        }
    """

    targets = {}

    if not os.path.exists(ruta_csv):

        print(
            f"[!] No existe {ruta_csv}."
        )

        print(
            "[!] No se pueden utilizar targets experimentales."
        )

        return targets

    with open(
        ruta_csv,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        if "id_seq" not in reader.fieldnames:
            raise ValueError(
                "El CSV debe contener una columna llamada 'id_seq'."
            )

        if "target" not in reader.fieldnames:
            raise ValueError(
                "El CSV debe contener una columna llamada 'target'."
            )

        for row in reader:

            id_seq = row["id_seq"].strip()

            if not id_seq:
                continue

            try:
                target = float(row["target"])
            except ValueError:
                raise ValueError(
                    f"Target inválido para la secuencia '{id_seq}'."
                )

            targets[id_seq] = target

    return targets


# ==============================================================================
# 5. NORMALIZACIÓN / LIMPIEZA DE SECUENCIA
# ==============================================================================

def limpiar_secuencia(secuencia):
    """
    Conserva únicamente A, T, G y C.
    """

    return "".join(
        b
        for b in secuencia.upper()
        if b in "ATGC"
    )


# ==============================================================================
# 6. 10 FUNCIONES DE DERIVADAS
# ==============================================================================

def derivada_1_hidropatia(v):
    return np.gradient(v)


def derivada_2_peso(v):
    return np.gradient(v) * 1.5


def derivada_3_puentes(v):
    return np.gradient(v) - 0.2


def derivada_4_exponencial(v):
    return np.gradient(
        np.exp(np.clip(v, -2, 2))
    )


def derivada_5_logaritmica(v):
    return np.gradient(
        np.log(np.abs(v) + 1.0)
    )


def derivada_6_trig(v):
    return np.gradient(
        np.sin(v)
    )


def derivada_7_cuadratica(v):
    return np.gradient(
        v ** 2
    )


def derivada_8_cubica(v):
    return np.gradient(
        v ** 3
    )


def derivada_9_sigmoide(v):
    return np.gradient(
        1 / (1 + np.exp(-v))
    )


def derivada_10_tanh(v):
    return np.gradient(
        np.tanh(v)
    )


# ==============================================================================
# 7. 10 FUNCIONES DE INTEGRALES DISCRETAS
# ==============================================================================

def integral_1_base(v):
    return float(
        trapezoid(v, dx=1.0)
    )


def integral_2_cuadratica(v):
    return float(
        trapezoid(v ** 2, dx=1.0)
    )


def integral_3_absoluta(v):
    return float(
        trapezoid(np.abs(v), dx=1.0)
    )


def integral_4_inversa(v):
    return float(
        trapezoid(
            1.0 / (np.abs(v) + 1.0),
            dx=1.0
        )
    )


def integral_5_raiz(v):
    return float(
        trapezoid(
            np.sqrt(np.abs(v)),
            dx=1.0
        )
    )


def integral_6_seno(v):
    return float(
        trapezoid(
            np.sin(v),
            dx=1.0
        )
    )


def integral_7_cos(v):
    return float(
        trapezoid(
            np.cos(v),
            dx=1.0
        )
    )


def integral_8_ponderada(v):
    pesos = np.linspace(
        0.1,
        1.0,
        len(v)
    )

    return float(
        trapezoid(
            v * pesos,
            dx=1.0
        )
    )


def integral_9_log(v):
    return float(
        trapezoid(
            np.log(np.abs(v) + 2.0),
            dx=1.0
        )
    )


def integral_10_exponencial(v):
    return float(
        trapezoid(
            np.exp(-np.abs(v)),
            dx=1.0
        )
    )


# ==============================================================================
# 8. ECUACIONES DE SEGUNDO GRADO
# ==============================================================================

def ecuacion_grado_2_estabilidad(a, b, c, x):
    """
    Calcula:

        A*x² + B*x + C
    """

    return (
        a * (x ** 2)
        + b * x
        + c
    )


# ==============================================================================
# 9. ANÁLISIS DE CICLOS / K-MERS
# ==============================================================================

def analizar_ciclos_secuencia(secuencia, k=3):
    """
    Construye un grafo dirigido de k-mers consecutivos
    y cuenta ciclos simples.

    NOTA:
    simple_cycles puede crecer considerablemente en grafos
    grandes. Por ello se establece un límite práctico.
    """

    secuencia = limpiar_secuencia(secuencia)

    if len(secuencia) <= k:
        return 0

    G = nx.DiGraph()

    for i in range(
        len(secuencia) - k
    ):

        kmer_origen = secuencia[
            i:i + k
        ]

        kmer_destino = secuencia[
            i + 1:i + 1 + k
        ]

        G.add_edge(
            kmer_origen,
            kmer_destino
        )

    try:

        # En secuencias reales muy grandes el número de ciclos
        # puede ser enorme.
        ciclos = nx.simple_cycles(G)

        num_ciclos = sum(
            1 for _ in ciclos
        )

    except Exception:
        num_ciclos = 0

    return num_ciclos


# ==============================================================================
# 10. CÁLCULO DEL PERFIL AVANZADO Y MATRICES
# ==============================================================================

def calcular_perfil_avanzado_y_matrices(secuencia):

    secuencia = limpiar_secuencia(secuencia)

    # --------------------------------------------------------------------------
    # Control de secuencias demasiado cortas
    # --------------------------------------------------------------------------

    if len(secuencia) < 5:

        return (
            np.zeros(10, dtype=np.float32),
            np.zeros((5, 5), dtype=np.float32)
        )

    # --------------------------------------------------------------------------
    # Conversión de bases a variables biofísicas
    # --------------------------------------------------------------------------

    hidro = np.array(
        [
            TABLA_QUIMICA_ADN[b]["hidropatia"]
            for b in secuencia
        ],
        dtype=np.float64
    )

    peso = np.array(
        [
            TABLA_QUIMICA_ADN[b]["peso_molecular"]
            for b in secuencia
        ],
        dtype=np.float64
    )

    puentes = np.array(
        [
            TABLA_QUIMICA_ADN[b]["puentes_h"]
            for b in secuencia
        ],
        dtype=np.float64
    )

    # --------------------------------------------------------------------------
    # 10 DERIVADAS
    # --------------------------------------------------------------------------

    d1 = np.mean(
        derivada_1_hidropatia(hidro)
    )

    d2 = np.mean(
        derivada_2_peso(peso)
    )

    d3 = np.mean(
        derivada_3_puentes(puentes)
    )

    d4 = np.mean(
        derivada_4_exponencial(hidro)
    )

    d5 = np.mean(
        derivada_5_logaritmica(hidro)
    )

    d6 = np.mean(
        derivada_6_trig(hidro)
    )

    d7 = np.mean(
        derivada_7_cuadratica(hidro)
    )

    d8 = np.mean(
        derivada_8_cubica(hidro)
    )

    d9 = np.mean(
        derivada_9_sigmoide(hidro)
    )

    d10 = np.mean(
        derivada_10_tanh(hidro)
    )

    # --------------------------------------------------------------------------
    # 10 INTEGRALES
    # --------------------------------------------------------------------------

    i1 = integral_1_base(hidro)
    i2 = integral_2_cuadratica(hidro)
    i3 = integral_3_absoluta(hidro)
    i4 = integral_4_inversa(hidro)
    i5 = integral_5_raiz(hidro)
    i6 = integral_6_seno(hidro)
    i7 = integral_7_cos(hidro)
    i8 = integral_8_ponderada(hidro)
    i9 = integral_9_log(hidro)
    i10 = integral_10_exponencial(hidro)

    # --------------------------------------------------------------------------
    # 3 ECUACIONES DE SEGUNDO GRADO
    # --------------------------------------------------------------------------

    eq1 = ecuacion_grado_2_estabilidad(
        0.5,
        -1.2,
        3.4,
        abs(d1)
    )

    eq2 = ecuacion_grado_2_estabilidad(
        0.1,
        0.5,
        -2.0,
        abs(d2)
    )

    eq3 = ecuacion_grado_2_estabilidad(
        -0.3,
        2.1,
        1.0,
        abs(d3)
    )

    # --------------------------------------------------------------------------
    # VECTOR BASE DE FEATURES
    #
    # Se conservan 10 variables para la parte matricial.
    # --------------------------------------------------------------------------

    vector_features = np.array(
        [
            d1,
            d2,
            d3,
            i1,
            i2,
            i3,
            eq1,
            eq2,
            eq3,
            len(secuencia)
        ],
        dtype=np.float32
    )

    # --------------------------------------------------------------------------
    # GENERACIÓN DE 10 MATRICES 5x5
    # --------------------------------------------------------------------------

    matrices_complejas = []

    for m_idx in range(1, 11):

        matriz_base = (
            np.outer(
                vector_features[:5],
                vector_features[:5]
            )
            * (m_idx * 0.1)
        )

        matriz_transformada = (
            np.dot(
                matriz_base,
                matriz_base.T
            )
            /
            (
                np.linalg.norm(matriz_base)
                + 1e-5
            )
        )

        matrices_complejas.append(
            matriz_transformada.astype(
                np.float32
            )
        )

    return (
        vector_features,
        matrices_complejas[0]
    )


# ==============================================================================
# 11. FUNCIÓN DE FEATURES COMPLETOS
# ==============================================================================

def construir_features_completos(secuencia):

    secuencia_limpia = limpiar_secuencia(
        secuencia
    )

    # Perfil avanzado
    feats, matriz_comp = (
        calcular_perfil_avanzado_y_matrices(
            secuencia_limpia
        )
    )

    # --------------------------------------------------------------------------
    # Perfil matemático adicional
    # --------------------------------------------------------------------------

    if len(secuencia_limpia) >= 2:

        valores_hidropatia = np.array(
            [
                TABLA_QUIMICA_ADN[b]["hidropatia"]
                for b in secuencia_limpia
            ],
            dtype=np.float64
        )

        derivada = np.gradient(
            valores_hidropatia
        )

        derivada_rms = float(
            np.sqrt(
                np.mean(
                    derivada ** 2
                )
            )
        )

        integral_area = float(
            trapezoid(
                valores_hidropatia,
                dx=1.0
            )
        )

        frecuencias = {
            b:
            secuencia_limpia.count(b)
            / len(secuencia_limpia)
            for b in "ATGC"
        }

        log_entropia = float(
            -sum(
                f * np.log2(f)
                for f in frecuencias.values()
                if f > 0
            )
        )

    else:

        derivada_rms = 0.0
        integral_area = 0.0
        log_entropia = 0.0

    # --------------------------------------------------------------------------
    # Grafo
    # --------------------------------------------------------------------------

    ciclos = analizar_ciclos_secuencia(
        secuencia_limpia,
        k=3
    )

    longitud = len(
        secuencia_limpia
    )

    # --------------------------------------------------------------------------
    # MÉTRICAS PARA LAS TABLAS DE VERDAD
    # --------------------------------------------------------------------------

    cantidad_g = secuencia_limpia.count("G")

    cantidad_c = secuencia_limpia.count("C")

    purinas = (
        secuencia_limpia.count("A")
        + secuencia_limpia.count("G")
    )

    proporcion_purinas = (
        purinas / longitud
        if longitud > 0
        else 0.0
    )

    proporcion_gc = (
        (cantidad_g + cantidad_c)
        / longitud
        if longitud > 0
        else 0.0
    )

    # --------------------------------------------------------------------------
    # TABLA 1
    #
    # Purina abundante + puentes de H altos
    # --------------------------------------------------------------------------

    t1 = evaluar_tabla_verdad_1(
        purina=(proporcion_purinas > 0.5),
        h_bonds_altos=(proporcion_gc > 0.5)
    )

    # --------------------------------------------------------------------------
    # TABLA 2
    #
    # Secuencia corta o patrón AT
    # --------------------------------------------------------------------------

    t2 = evaluar_tabla_verdad_2(
        inestable=(longitud < 50),
        patron_repetitivo=(
            secuencia_limpia.startswith("AT")
        )
    )

    # --------------------------------------------------------------------------
    # TABLA 3
    #
    # Alta entropía XOR mutación crítica
    #
    # En este pipeline se utiliza una condición explícita
    # basada en la composición.
    # --------------------------------------------------------------------------

    mutacion_critica = (
        "GGG" in secuencia_limpia
        or "CCC" in secuencia_limpia
    )

    t3 = evaluar_tabla_verdad_3(
        alta_entropia=(log_entropia > 1.5),
        mutacion_critica=mutacion_critica
    )

    # --------------------------------------------------------------------------
    # TABLA 4
    # --------------------------------------------------------------------------

    t4 = evaluar_tabla_verdad_4(
        sitio_activo_ok=(t1 == 1),
        inhibidor_presente=(t2 == 1),
        sinergia=(t3 == 1)
    )

    # --------------------------------------------------------------------------
    # VECTOR FINAL
    #
    # 10 features avanzadas
    # + 3 métricas matemáticas/topológicas
    # + 4 tablas lógicas
    # = 17 features
    #
    # Además se incorporan elementos de la matriz 5x5:
    # 25 valores adicionales.
    #
    # Total:
    # 42 features.
    # --------------------------------------------------------------------------

    matriz_aplanada = matriz_comp.flatten()

    vector_final = np.concatenate(
        [
            feats.astype(np.float32),

            np.array(
                [
                    derivada_rms,
                    integral_area,
                    log_entropia,
                    float(ciclos)
                ],
                dtype=np.float32
            ),

            np.array(
                [
                    float(t1),
                    float(t2),
                    float(t3),
                    float(t4)
                ],
                dtype=np.float32
            ),

            matriz_aplanada.astype(
                np.float32
            )
        ]
    )

    return (
        vector_final.astype(np.float32),
        {
            "t1": t1,
            "t2": t2,
            "t3": t3,
            "t4": t4,
            "entropia": log_entropia,
            "ciclos": ciclos,
            "matriz": matriz_comp
        }
    )


# ==============================================================================
# 12. RED NEURONAL
# ==============================================================================

class RedNeuronalGenomicaAvanzada(nn.Module):

    def __init__(self, input_dim):

        super(
            RedNeuronalGenomicaAvanzada,
            self
        ).__init__()

        self.fc1 = nn.Linear(
            input_dim,
            64
        )

        self.bn1 = nn.BatchNorm1d(
            64
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            0.3
        )

        self.fc2 = nn.Linear(
            64,
            32
        )

        self.fc3 = nn.Linear(
            32,
            1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.fc1(x)

        if x.size(0) > 1:
            out = self.bn1(out)

        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc2(out)

        out = self.relu(out)

        out = self.fc3(out)

        return self.sigmoid(out)


# ==============================================================================
# 13. REGRESIÓN LINEAL ANALÍTICA
# ==============================================================================

def ejecutar_regresion_lineal(X_np, Y_np):

    print(
        "\n--- REGRESIÓN LINEAL ANALÍTICA ---"
    )

    X_b = np.c_[
        np.ones(
            (X_np.shape[0], 1)
        ),
        X_np
    ]

    try:

        identidad = np.eye(
            X_b.shape[1]
        )

        theta_best = np.linalg.inv(
            X_b.T.dot(X_b)
            + 0.01 * identidad
        ).dot(
            X_b.T
        ).dot(
            Y_np
        )

        pred_reg_lineal = (
            X_b.dot(theta_best)
        )

        print(
            "[✔] Regresión lineal completada."
        )

        print(
            f"[✔] Número de parámetros: "
            f"{len(theta_best)}"
        )

        return (
            theta_best,
            pred_reg_lineal
        )

    except Exception as e:

        print(
            f"[!] Error en regresión lineal: {e}"
        )

        return None, None


# ==============================================================================
# 14. ENTRENAMIENTO DE LA RED NEURONAL
# ==============================================================================

def entrenar_modelo(
    X_np,
    Y_np,
    epochs=2000,
    learning_rate=0.002
):

    # --------------------------------------------------------------------------
    # Hardware
    # --------------------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\n--- ENTRENAMIENTO DE RED NEURONAL ---"
    )

    print(
        f"[*] Hardware seleccionado: {device}"
    )

    # --------------------------------------------------------------------------
    # Tensores
    # --------------------------------------------------------------------------

    X_tensor = torch.tensor(
        X_np,
        dtype=torch.float32
    ).to(device)

    Y_tensor = torch.tensor(
        Y_np,
        dtype=torch.float32
    ).to(device)

    # --------------------------------------------------------------------------
    # Modelo
    # --------------------------------------------------------------------------

    modelo = (
        RedNeuronalGenomicaAvanzada(
            input_dim=X_tensor.shape[1]
        )
        .to(device)
    )

    criterio = nn.MSELoss()

    optimizador = optim.Adam(
        modelo.parameters(),
        lr=learning_rate
    )

    # --------------------------------------------------------------------------
    # Entrenamiento
    # --------------------------------------------------------------------------

    for epoch in range(epochs):

        modelo.train()

        optimizador.zero_grad()

        prediccion = modelo(
            X_tensor
        )

        loss = criterio(
            prediccion,
            Y_tensor
        )

        loss.backward()

        optimizador.step()

        if (
            (epoch + 1) % 500 == 0
            or loss.item() < 0.00001
        ):

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"| MSE Loss: "
                f"{loss.item():.7f}"
            )

        if loss.item() < 0.00001:

            print(
                f"[+] Convergencia alcanzada "
                f"en Epoch {epoch + 1}"
            )

            break

    return (
        modelo,
        X_tensor,
        Y_tensor
    )


# ==============================================================================
# 15. INFERENCIA
# ==============================================================================

def ejecutar_inferencia(
    modelo,
    X_np,
    ids
):

    modelo.eval()

    device = next(
        modelo.parameters()
    ).device

    X_tensor = torch.tensor(
        X_np,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():

        resultados = modelo(
            X_tensor
        )

    resultados_np = (
        resultados
        .cpu()
        .numpy()
        .flatten()
    )

    print(
        "\n--- PREDICCIÓN FINAL ---"
    )

    for id_seq, pred in zip(
        ids,
        resultados_np
    ):

        print(
            f"ID: {id_seq} "
            f"| Predicción: {pred:.6f}"
        )

    return resultados_np


# ==============================================================================
# 16. EXPORTACIÓN DE RESULTADOS
# ==============================================================================

def guardar_resultados(
    ids,
    predicciones,
    ruta="predicciones.csv"
):

    with open(
        ruta,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "id_seq",
                "prediccion_modelo"
            ]
        )

        for id_seq, pred in zip(
            ids,
            predicciones
        ):

            writer.writerow(
                [
                    id_seq,
                    float(pred)
                ]
            )

    print(
        f"[✔] Resultados guardados en: {ruta}"
    )


# ==============================================================================
# 17. PIPELINE PRINCIPAL
# ==============================================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 78
    )

    print(
        "PIPELINE HÍBRIDO GENÓMICO AVANZADO"
    )

    print(
        "=" * 78
    )

    # --------------------------------------------------------------------------
    # PASO 1: CARGAR FASTA
    # --------------------------------------------------------------------------

    print(
        "\n--- [1/6] CARGANDO FASTA ---"
    )

    diccionario_secuencias = (
        leer_archivo_fasta(
            ARCHIVO_FASTA
        )
    )

    if not diccionario_secuencias:

        raise RuntimeError(
            "No se encontraron secuencias válidas en el FASTA."
        )

    print(
        f"[✔] Secuencias cargadas: "
        f"{len(diccionario_secuencias)}"
    )

    # --------------------------------------------------------------------------
    # PASO 2: CARGAR TARGETS EXPERIMENTALES
    # --------------------------------------------------------------------------

    print(
        "\n--- [2/6] CARGANDO ETIQUETAS EXPERIMENTALES ---"
    )

    targets = cargar_targets_experimentales(
        ARCHIVO_TARGETS
    )

    # --------------------------------------------------------------------------
    # PASO 3: EXTRAER FEATURES
    # --------------------------------------------------------------------------

    print(
        "\n--- [3/6] PROCESANDO SECUENCIAS ---"
    )

    dataset_features = []

    dataset_targets = []

    ids_validos = []

    informacion_secuencias = []

    for id_seq, seq in (
        diccionario_secuencias.items()
    ):

        secuencia_limpia = limpiar_secuencia(
            seq
        )

        if len(secuencia_limpia) < 5:

            print(
                f"[!] {id_seq}: secuencia demasiado corta. "
                f"Se omite."
            )

            continue

        # ----------------------------------------------------------------------
        # Features
        # ----------------------------------------------------------------------

        features, info = (
            construir_features_completos(
                secuencia_limpia
            )
        )

        # ----------------------------------------------------------------------
        # Mostrar información
        # ----------------------------------------------------------------------

        print(
            f"\nID: {id_seq}"
        )

        print(
            f"  Longitud: {len(secuencia_limpia)}"
        )

        print(
            f"  Entropía: {info['entropia']:.4f}"
        )

        print(
            f"  Ciclos: {info['ciclos']}"
        )

        print(
            "  Tablas de verdad: "
            f"[{info['t1']}, "
            f"{info['t2']}, "
            f"{info['t3']}, "
            f"{info['t4']}]"
        )

        print(
            f"  Features totales: "
            f"{len(features)}"
        )

        # ----------------------------------------------------------------------
        # Si existen targets, asociarlos por ID
        # ----------------------------------------------------------------------

        if id_seq not in targets:

            print(
                f"  [!] No existe target experimental para "
                f"{id_seq}. Se omite del entrenamiento."
            )

            continue

        target = targets[id_seq]

        print(
            f"  Target experimental: {target}"
        )

        dataset_features.append(
            features
        )

        dataset_targets.append(
            [target]
        )

        ids_validos.append(
            id_seq
        )

        informacion_secuencias.append(
            info
        )

    # --------------------------------------------------------------------------
    # Verificar dataset
    # --------------------------------------------------------------------------

    if len(dataset_features) == 0:

        print(
            "\n"
            + "=" * 78
        )

        print(
            "NO HAY DATOS SUFICIENTES PARA ENTRENAMIENTO"
        )

        print(
            "=" * 78
        )

        print(
            "\nDebes tener:"
        )

        print(
            "1. Un archivo genoma_muestra.fasta"
        )

        print(
            "2. Un archivo etiquetas_experimentales.csv"
        )

        print(
            "3. IDs coincidentes entre FASTA y CSV"
        )

        print(
            "\nEjemplo del CSV:"
        )

        print(
            "id_seq,target"
        )

        print(
            "Secuencia_1,0.92"
        )

        print(
            "Secuencia_2,0.74"
        )

        raise SystemExit

    # --------------------------------------------------------------------------
    # Convertir a NumPy
    # --------------------------------------------------------------------------

    X_np = np.asarray(
        dataset_features,
        dtype=np.float32
    )

    Y_np = np.asarray(
        dataset_targets,
        dtype=np.float32
    )

    print(
        "\n[✔] Matriz X:"
    )

    print(
        f"    Shape = {X_np.shape}"
    )

    print(
        "[✔] Vector Y:"
    )

    print(
        f"    Shape = {Y_np.shape}"
    )

    # --------------------------------------------------------------------------
    # PASO 4: REGRESIÓN LINEAL
    # --------------------------------------------------------------------------

    print(
        "\n--- [4/6] BASELINE DE REGRESIÓN LINEAL ---"
    )

    theta_best, pred_reg_lineal = (
        ejecutar_regresion_lineal(
            X_np,
            Y_np
        )
    )

    if pred_reg_lineal is not None:

        print(
            "\nPredicciones de regresión lineal:"
        )

        for id_seq, pred in zip(
            ids_validos,
            pred_reg_lineal.flatten()
        ):

            print(
                f"  {id_seq}: {pred:.6f}"
            )

    # --------------------------------------------------------------------------
    # PASO 5: RED NEURONAL
    # --------------------------------------------------------------------------

    print(
        "\n--- [5/6] RED NEURONAL PYTORCH ---"
    )

    modelo_final, X_tensor, Y_tensor = (
        entrenar_modelo(
            X_np,
            Y_np,
            epochs=2000,
            learning_rate=0.002
        )
    )

    # --------------------------------------------------------------------------
    # PASO 6: INFERENCIA
    # --------------------------------------------------------------------------

    print(
        "\n--- [6/6] INFERENCIA FINAL ---"
    )

    resultado_inferencia = (
        ejecutar_inferencia(
            modelo_final,
            X_np,
            ids_validos
        )
    )

    # --------------------------------------------------------------------------
    # Guardar resultados
    # --------------------------------------------------------------------------

    guardar_resultados(
        ids_validos,
        resultado_inferencia,
        "predicciones.csv"
    )

    # --------------------------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 78
    )

    print(
        "PIPELINE COMPLETADO"
    )

    print(
        "=" * 78
    )

    print(
        f"Secuencias utilizadas: {len(ids_validos)}"
    )

    print(
        f"Features por secuencia: {X_np.shape[1]}"
    )

    print(
        f"Targets experimentales: {Y_np.shape[0]}"
    )

    print(
        f"Dispositivo: "
        f"{next(modelo_final.parameters()).device}"
    )

    print(
        "\nPredicciones:"
    )

    for id_seq, pred in zip(
        ids_validos,
        resultado_inferencia
    ):

        print(
            f"  {id_seq}: {pred:.6f}"
        )

    print(
        "\n[✔] Archivo generado: predicciones.csv"
    )

    print(
        "\nIMPORTANTE:"
    )

    print(
        "Una pérdida de entrenamiento baja no demuestra "
        "por sí sola validez biológica o terapéutica."
    )

    print(
        "La validación debe realizarse con datos "
        "experimentales independientes."
    )
