# ==============================================================================
# PIPELINE HÍBRIDO AVANZADO V1.0.5: ARQUITECTURA MULTIMODAL CON ESPACIO LATENTE Z
# Y ANÁLISIS MATEMÁTICO-TOPOLÓGICO COMPLETO 
# Basado en la especificación formal y evolución de Nadal Ferrà (2026)
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
# 0. CONFIGURACIÓN Y BIOFÍSICA BÁSICA
# ==============================================================================

ARCHIVO_FASTA = "genoma_muestra.fasta"
ARCHIVO_TARGETS = "etiquetas_experimentales.csv"
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

TABLA_QUIMICA_ADN = {
    'A': {'peso_molecular': 135.13, 'hidropatia': -1.9, 'puentes_h': 2, 'tipo': 'Purina'},
    'T': {'peso_molecular': 126.11, 'hidropatia': -0.7, 'puentes_h': 2, 'tipo': 'Pirimidina'},
    'G': {'peso_molecular': 151.13, 'hidropatia': -3.5, 'puentes_h': 3, 'tipo': 'Purina'},
    'C': {'peso_molecular': 111.10, 'hidropatia': -3.5, 'puentes_h': 3, 'tipo': 'Pirimidina'}
}

# ==============================================================================
# 1. TABLAS DE VERDAD LÓGICO-GENÓMICAS
# ==============================================================================

def evaluar_tabla_verdad_1(purina, h_bonds_altos):
    """Regulación de unión a receptor (Compuerta AND)."""
    return int(bool(purina) and bool(h_bonds_altos))

def evaluar_tabla_verdad_2(inestable, patron_repetitivo):
    """Alerta de mutación de escape (Compuerta OR)."""
    return int(bool(inestable) or bool(patron_repetitivo))

def evaluar_tabla_verdad_3(alta_entropia, mutacion_critica):
    """Resistencia antimicrobiana condicional (Compuerta XOR)."""
    return int(bool(alta_entropia)) ^ int(bool(mutacion_critica))

def evaluar_tabla_verdad_4(sitio_activo_ok, inhibidor_presente, sinergia):
    """Eficacia terapéutica binaria combinada: (A OR B) AND NOT C."""
    return int((bool(sitio_activo_ok) or bool(inhibidor_presente)) and not bool(sinergia))

# ==============================================================================
# 2. PARSER FASTA Y CARGA DE TARGETS EXPERIMENTALES
# ==============================================================================

def leer_archivo_fasta(ruta_archivo):
    secuencias = {}
    if not os.path.exists(ruta_archivo):
        print(f"[!] No se encontró {ruta_archivo}. Usando secuencia por defecto.")
        return {"Secuencia_Defecto": "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGA"}
    
    current_header = None
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_header = line[1:].strip()
                secuencias[current_header] = ""
            elif current_header is not None:
                secuencias[current_header] += line.upper()
    return secuencias

def cargar_targets_experimentales(ruta_csv):
    targets = {}
    if not os.path.exists(ruta_csv):
        print(f"[!] No existe {ruta_csv}. Se utilizarán targets simulados por defecto.")
        return targets

    with open(ruta_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if "id_seq" not in reader.fieldnames or "target" not in reader.fieldnames:
            raise ValueError("El CSV debe contener las columnas 'id_seq' y 'target'.")
        for row in reader:
            id_seq = row["id_seq"].strip()
            if id_seq:
                targets[id_seq] = float(row["target"])
    return targets

def limpiar_secuencia(secuencia):
    return "".join([b for b in secuencia.upper() if b in "ATGC"])

# ==============================================================================
# 3. MÓDULO MATEMÁTICO:  DERIVADAS Y  INTEGRALES 
# ==============================================================================

def derivada_1_hidropatia(v): return np.gradient(v)
def derivada_2_peso(v): return np.gradient(v) * 1.5
def derivada_3_puentes(v): return np.gradient(v) - 0.2
def derivada_4_exponencial(v): return np.gradient(np.exp(np.clip(v, -2, 2)))
def derivada_5_logaritmica(v): return np.gradient(np.log(np.abs(v) + 1.0))
def derivada_6_trig(v): return np.gradient(np.sin(v))
def derivada_7_cuadratica(v): return np.gradient(v ** 2)
def derivada_8_cubica(v): return np.gradient(v ** 3)
def derivada_9_sigmoide(v): return np.gradient(1 / (1 + np.exp(-v)))
def derivada_10_tanh(v): return np.gradient(np.tanh(v))

def integral_1_base(v): return float(trapezoid(v, dx=1.0))
def integral_2_cuadratica(v): return float(trapezoid(v ** 2, dx=1.0))
def integral_3_absoluta(v): return float(trapezoid(np.abs(v), dx=1.0))
def integral_4_inversa(v): return float(trapezoid(1.0 / (np.abs(v) + 1.0), dx=1.0))
def integral_5_raiz(v): return float(trapezoid(np.sqrt(np.abs(v)), dx=1.0))
def integral_6_seno(v): return float(trapezoid(np.sin(v), dx=1.0))
def integral_7_cos(v): return float(trapezoid(np.cos(v), dx=1.0))
def integral_8_ponderada(v): return float(trapezoid(v * np.linspace(0.1, 1.0, len(v)), dx=1.0))
def integral_9_log(v): return float(trapezoid(np.log(np.abs(v) + 2.0), dx=1.0))
def integral_10_exponencial(v): return float(trapezoid(np.exp(-np.abs(v)), dx=1.0))

def ecuacion_grado_2_estabilidad(a, b, c, x):
    return a * (x ** 2) + b * x + c

def analizar_ciclos_secuencia(secuencia, k=3):
    secuencia = limpiar_secuencia(secuencia)
    if len(secuencia) <= k:
        return 0
    G = nx.DiGraph()
    for i in range(len(secuencia) - k):
        G.add_edge(secuencia[i:i+k], secuencia[i+1:i+1+k])
    try:
        return sum(1 for _ in nx.simple_cycles(G))
    except Exception:
        return 0

def calcular_perfil_matematico_avanzado(secuencia):
    secuencia = limpiar_secuencia(secuencia)
    if len(secuencia) < 5:
        return np.zeros(10, dtype=np.float32), np.zeros((5, 5), dtype=np.float32)

    hidro = np.array([TABLA_QUIMICA_ADN[b]["hidropatia"] for b in secuencia], dtype=np.float64)
    peso = np.array([TABLA_QUIMICA_ADN[b]["peso_molecular"] for b in secuencia], dtype=np.float64)
    puentes = np.array([TABLA_QUIMICA_ADN[b]["puentes_h"] for b in secuencia], dtype=np.float64)

    d1, d2, d3 = np.mean(derivada_1_hidropatia(hidro)), np.mean(derivada_2_peso(peso)), np.mean(derivada_3_puentes(puentes))
    d4, d5, d6 = np.mean(derivada_4_exponencial(hidro)), np.mean(derivada_5_logaritmica(hidro)), np.mean(derivada_6_trig(hidro))
    d7, d8, d9, d10 = np.mean(derivada_7_cuadratica(hidro)), np.mean(derivada_8_cubica(hidro)), np.mean(derivada_9_sigmoide(hidro)), np.mean(derivada_10_tanh(hidro))

    i1, i2, i3, i4, i5 = integral_1_base(hidro), integral_2_cuadratica(hidro), integral_3_absoluta(hidro), integral_4_inversa(hidro), integral_5_raiz(hidro)
    i6, i7, i8, i9, i10 = integral_6_seno(hidro), integral_7_cos(hidro), integral_8_ponderada(hidro), integral_9_log(hidro), integral_10_exponencial(hidro)

    eq1 = ecuacion_grado_2_estabilidad(0.5, -1.2, 3.4, abs(d1))
    eq2 = ecuacion_grado_2_estabilidad(0.1, 0.5, -2.0, abs(d2))
    eq3 = ecuacion_grado_2_estabilidad(-0.3, 2.1, 1.0, abs(d3))

    vector_features = np.array([d1, d2, d3, i1, i2, i3, eq1, eq2, eq3, len(secuencia)], dtype=np.float32)
    
    matriz_base = np.outer(vector_features[:5], vector_features[:5]) * 0.1
    matriz_transformada = np.dot(matriz_base, matriz_base.T) / (np.linalg.norm(matriz_base) + 1e-5)

    return vector_features, matriz_transformada.astype(np.float32)

# ==============================================================================
# 4. EXTRACCIÓN DE MODALIDADES HETEROGÉNEAS (DDNA, Dspatial, Dstrat)
# ==============================================================================

def extraer_modalidades_multimodales(secuencia):
    secuencia = limpiar_secuencia(secuencia)
    if len(secuencia) < 5:
        secuencia = "ATGCGATCGATC"

    longitud = len(secuencia)
    freqs = [secuencia.count(b) / longitud for b in "ATGC"]
    hidros = [TABLA_QUIMICA_ADN[b]['hidropatia'] for b in secuencia]
    pesos = [TABLA_QUIMICA_ADN[b]['peso_molecular'] for b in secuencia]
    
    # 1. Modalidad Genómica (DDNA - 16 dims)
    ddna = freqs + [np.mean(hidros), np.std(hidros), np.mean(pesos), np.std(pesos)] + [0.0] * 8
    ddna = ddna[:16]

    # 2. Modalidad Geométrica (Dspatial - 4 dims)
    ciclos = analizar_ciclos_secuencia(secuencia, k=3)
    G = nx.DiGraph()
    for i in range(len(secuencia) - 3):
        G.add_edge(secuencia[i:i+3], secuencia[i+1:i+1+3])
    dspatial = [float(ciclos), float(G.number_of_nodes()), float(G.number_of_edges()), np.linalg.norm(hidros) if len(hidros)>0 else 0.0]

    # 3. Modalidad Estratigráfica (Dstrat - 4 dims)
    derivada = np.gradient(hidros) if len(hidros) > 1 else np.array([0.0])
    derivada_rms = float(np.sqrt(np.mean(derivada**2)))
    integral_area = float(trapezoid(hidros, dx=1.0))
    log_entropia = float(-sum(f * np.log2(f) for f in freqs if f > 0))
    dstrat = [derivada_rms, integral_area, log_entropia, float(longitud)]

    # 4. Evaluación de Tablas de Verdad Lógicas
    purinas = (secuencia.count("A") + secuencia.count("G")) / longitud
    gc_ratio = (secuencia.count("G") + secuencia.count("C")) / longitud
    t1 = evaluar_tabla_verdad_1(purinas > 0.5, gc_ratio > 0.5)
    t2 = evaluar_tabla_verdad_2(longitud < 50, secuencia.startswith("AT"))
    t3 = evaluar_tabla_verdad_3(log_entropia > 1.5, "GGG" in secuencia or "CCC" in secuencia)
    t4 = evaluar_tabla_verdad_4(t1 == 1, t2 == 1, t3 == 1)
    
    tablas_logicas = [float(t1), float(t2), float(t3), float(t4)]

    # Obtenemos perfil matemático avanzado extendido
    feats_adv, matriz_comp = calcular_perfil_matematico_avanzado(secuencia)
    
    return (
        np.array(ddna, dtype=np.float32), 
        np.array(dspatial, dtype=np.float32), 
        np.array(dstrat, dtype=np.float32),
        np.array(tablas_logicas, dtype=np.float32),
        feats_adv,
        matriz_comp.flatten()
    )

# ==============================================================================
# 5. ARQUITECTURA MULTIMODAL CON ESPACIO LATENTE Z (NADAL FERRÀ, 2026)
# ==============================================================================

class EncodersMultimodalesYZ(nn.Module):
    def __init__(self, p_dna=16, p_spatial=4, p_strat=4, d1=16, d2=16, d3=16, dz=32):
        super(EncodersMultimodalesYZ, self).__init__()
        
        self.encoder_dna = nn.Sequential(
            nn.Linear(p_dna, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, d1)
        )
        
        self.encoder_spatial = nn.Sequential(
            nn.Linear(p_spatial, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, d2)
        )
        
        self.encoder_strat = nn.Sequential(
            nn.Linear(p_strat, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, d3)
        )
        
        # Fusión Tardía Proyectada hacia el Espacio Latente Z
        self.W1 = nn.Parameter(torch.randn(dz, d1) * 0.02)
        self.W2 = nn.Parameter(torch.randn(dz, d2) * 0.02)
        self.W3 = nn.Parameter(torch.randn(dz, d3) * 0.02)
        self.bias_z = nn.Parameter(torch.zeros(dz))
        
        self.activacion_z = nn.GELU()
        
        # Predictor posterior g_theta enriquecido con métricas históricas
        self.predictor = nn.Sequential(
            nn.Linear(dz + 4 + 10 + 25, 64), # Latente Z + Tablas Lógicas + Features Adv + Matriz 5x5 aplanada
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x_dna, x_spatial, x_strat, x_logicas, x_avanzados, x_matriz, mask=None):
        if mask is not None:
            x_dna = x_dna * mask[:, 0:1]
            x_spatial = x_spatial * mask[:, 1:2]
            x_strat = x_strat * mask[:, 2:3]

        e_dna = self.encoder_dna(x_dna)
        e_spatial = self.encoder_spatial(x_spatial)
        e_strat = self.encoder_strat(x_strat)

        term1 = torch.matmul(e_dna, self.W1.T)
        term2 = torch.matmul(e_spatial, self.W2.T)
        term3 = torch.matmul(e_strat, self.W3.T)
        
        Z = self.activacion_z(term1 + term2 + term3 + self.bias_z)

        # Concatenación con el vector histórico integral expandido
        features_completos = torch.cat([Z, x_logicas, x_avanzados, x_matriz], dim=1)
        y_hat = self.predictor(features_completos)
        
        return y_hat, Z

# ==============================================================================
# 6. PIPELINE PRINCIPAL DE ENTRENAMIENTO Y VALIDACIÓN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*78)
    print("PIPELINE HÍBRIDO ÉPICO: MODELO GENÓMICO V1.0.5 CON ESPACIO LATENTE Z")
    print("="*78)

    # 1. Cargar Datos FASTA y Targets
    diccionario_secuencias = leer_archivo_fasta(ARCHIVO_FASTA)
    targets_dict = cargar_targets_experimentales(ARCHIVO_TARGETS)

    ids_validos = []
    t_dna, t_spatial, t_strat, t_logicas, t_avanzados, t_matriz, t_targets = [], [], [], [], [], [], []

    print("\n--- [1/4] EXTRACCIÓN DE MODALIDADES Y FEATURES HISTÓRICOS ---")
    for id_seq, seq in diccionario_secuencias.items():
        seq_limpia = limpiar_secuencia(seq)
        if len(seq_limpia) < 5:
            continue
        
        ddna, dspatial, dstrat, logicas, adv, matriz = extraer_modalidades_multimodales(seq_limpia)
        
        target = targets_dict.get(id_seq, 0.85) # Si no hay CSV, asigna valor base de prueba
        
        ids_validos.append(id_seq)
        t_dna.append(ddna)
        t_spatial.append(dspatial)
        t_strat.append(dstrat)
        t_logicas.append(logicas)
        t_avanzados.append(adv)
        t_matriz.append(matriz)
        t_targets.append([target])
        print(f"  [✔] Procesado ID: {id_seq} | Longitud: {len(seq_limpia)}")

    X_dna = torch.tensor(np.array(t_dna), dtype=torch.float32)
    X_spatial = torch.tensor(np.array(t_spatial), dtype=torch.float32)
    X_strat = torch.tensor(np.array(t_strat), dtype=torch.float32)
    X_logicas = torch.tensor(np.array(t_logicas), dtype=torch.float32)
    X_avanzados = torch.tensor(np.array(t_avanzados), dtype=torch.float32)
    X_matriz = torch.tensor(np.array(t_matriz), dtype=torch.float32)
    Y = torch.tensor(np.array(t_targets), dtype=torch.float32)

    # 2. Inicialización de Hardware y Modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- [2/4] INICIALIZANDO MODELO EN HARDWARE: {device} ---")
    
    modelo = EncodersMultimodalesYZ().to(device)
    criterio = nn.MSELoss()
    optimizador = optim.AdamW(modelo.parameters(), lr=0.002, weight_decay=1e-4)

    X_dna_d = X_dna.to(device)
    X_spatial_d = X_spatial.to(device)
    X_strat_d = X_strat.to(device)
    X_logicas_d = X_logicas.to(device)
    X_avanzados_d = X_avanzados.to(device)
    X_matriz_d = X_matriz.to(device)
    Y_d = Y.to(device)

    # 3. Entrenamiento con Optimización por Lotes Simulados
    print("\n--- [3/4] ENTRENAMIENTO DEL ESPACIO LATENTE Z Y MÓDULO PREDICTOR ---")
    epochs = 2000
    for epoch in range(epochs):
        modelo.train()
        optimizador.zero_grad()
        
        predicciones, Z_latente = modelo(X_dna_d, X_spatial_d, X_strat_d, X_logicas_d, X_avanzados_d, X_matriz_d)
        loss = criterio(predicciones, Y_d)
        
        loss.backward()
        optimizador.step()

        if (epoch + 1) % 500 == 0 or loss.item() < 0.00001:
            print(f"Época [{epoch+1}/{epochs}] | MSE Loss: {loss.item():.7f}")
        if loss.item() < 0.00001:
            print(f"[+] Convergencia lograda en época {epoch+1}")
            break

    # 4. Inferencia Final y Pruebas de Robustez
    print("\n--- [4/4] INFERENCIA FINAL Y TEST DE ROBUSTEZ MULTIMODAL ---")
    modelo.eval()
    with torch.no_grad():
        y_final, Z_representacion = modelo(X_dna_d, X_spatial_d, X_strat_d, X_logicas_d, X_avanzados_d, X_matriz_d)
        
        print(f"\n[✔] Dimensión del Espacio Latente Z: {Z_representacion.shape}")
        for i, id_s in enumerate(ids_validos):
            print(f"  - ID: {id_s} | Predicción Final: {y_final[i].item():.6f}")

        # Test de Robustez ante apagón de modalidad estratigráfica
        mask_ausencia = torch.ones((len(ids_validos), 3), device=device)
        mask_ausencia[:, 2] = 0.0 
        y_robusto, _ = modelo(X_dna_d, X_spatial_d, X_strat_d, X_logicas_d, X_avanzados_d, X_matriz_d, mask=mask_ausencia)
        
        print("\n[🛡] TEST DE ROBUSTEZ (Ausencia de Datos Estratigráficos):")
        for i, id_s in enumerate(ids_validos):
            print(f"  - ID: {id_s} | Predicción Degradada Controlada: {y_robusto[i].item():.6f}")

    print("\n" + "="*78)
    print("EJECUCIÓN  COMPLETADA EXITOSAMENTE")
    print("="*78)