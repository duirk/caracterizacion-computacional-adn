# ==============================================================================
# PIPELINE HÍBRIDO UNIFICADO V3.0.0: GENÓMICA CUÁNTICA (Q-MLPD) + SÍNTESIS GEOMÉTRICA EX NIHILO
# Integración de Arquitectura Multimodal con Membranas Latentes, Oráculo Cuántico 
# y Motor de Generación Visual Matemática Pura (Inspirado en Nadal Ferrà & Midjourney Studio)
# ==============================================================================

import os
import csv
import time
import numpy as np
import networkx as nx
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

try:
    import cv2
    CV2_DISPONIBLE = True
except ImportError:
    CV2_DISPONIBLE = False

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
# 1. FÓRMULAS PROPIAS LAMBDA (h) Y LÓGICA DIFUSA GENÓMICA
# ==============================================================================

class FactorLambda(nn.Module):
    """Encapsula las fórmulas propias lambda (h) como módulos matemáticos aprendibles[cite: 2]."""
    def __init__(self, num_features=10):
        super(FactorLambda, self).__init__()
        self.peso_lambda = nn.Parameter(torch.ones(num_features))

    def forward(self, vector_features):
        return vector_features * F.softplus(self.peso_lambda)

def evaluar_logica_difusa_genomica(purina_ratio, gc_ratio, log_entropia, len_val, startswith_at, has_triple):
    """Evalúa reglas de regulación biológica mediante lógica difusa continua[cite: 2]."""
    union_receptor = torch.sigmoid(torch.tensor(purina_ratio * gc_ratio * 5.0))
    inestable = torch.sigmoid(torch.tensor((50.0 - len_val) / 10.0))
    at_flag = 1.0 if startswith_at else 0.0
    mutacion_escape = torch.maximum(inestable, torch.tensor(at_flag))
    
    alta_entropia = torch.sigmoid(torch.tensor((log_entropia - 1.5) * 2.0))
    mutacion_critica = 1.0 if has_triple else 0.0
    resistencia = torch.abs(alta_entropia - torch.tensor(mutacion_critica))
    
    a_or_b = torch.maximum(union_receptor, mutacion_escape)
    not_c = 1.0 - resistencia
    eficacia = torch.minimum(a_or_b, not_c)
    
    return torch.stack([union_receptor, mutacion_escape, resistencia, eficacia])

# ==============================================================================
# 2. PARSER FASTA Y CARGA DE TARGETS EXPERIMENTALES
# ==============================================================================

def leer_archivo_fasta(ruta_archivo):
    secuencias = {}
    if not os.path.exists(ruta_archivo):
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
# 3. MÓDULO MATEMÁTICO: DERIVADAS, INTEGRALES Y TOPOLOGÍA DE GRAFOS
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
# 4. MOTOR DE SÍNTESIS GEOMÉTRICA PURA (EX NIHILO / FENOTIPO VISUAL)
# ==============================================================================

class GeneradorGeometriaPuraMatematica(nn.Module):
    """Genera geometría estructurada y campos armónicos directamente desde el espacio latente Z."""
    def __init__(self, width=256, height=256, dz=32):
        super(GeneradorGeometriaPuraMatematica, self).__init__()
        self.width = width
        self.height = height
        self.dz = dz
        
        self.sintetizador_espacial = nn.Sequential(
            nn.Linear(dz, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, width * height * 3),
            nn.Tanh()
        )

    def fisica_longitud_onda_a_rgb(self, lambda_nm):
        r, g, b = 0.0, 0.0, 0.0
        if 380 <= lambda_nm < 440:
            r = -(lambda_nm - 440) / (440 - 380)
            b = 1.0
        elif 440 <= lambda_nm < 490:
            g = (lambda_nm - 440) / (490 - 440)
            b = 1.0
        elif 490 <= lambda_nm < 510:
            g = 1.0
            b = -(lambda_nm - 510) / (510 - 490)
        elif 510 <= lambda_nm < 580:
            r = (lambda_nm - 510) / (580 - 510)
            g = 1.0
        elif 580 <= lambda_nm < 645:
            r = 1.0
            g = -(lambda_nm - 645) / (645 - 580)
        elif 645 <= lambda_nm <= 750:
            r = 1.0

        factor = 0.3 + 0.7 * (750 - lambda_nm) / (750 - 380) if 380 <= lambda_nm <= 750 else 0.0
        return np.clip(r * factor, 0.0, 1.0), np.clip(g * factor, 0.0, 1.0), np.clip(b * factor, 0.0, 1.0)

    def forward_generar_geometria(self, z_latente, seed=42):
        batch_size = z_latente.shape[0]
        tensor_plano = self.sintetizador_espacial(z_latente)
        canvas_tensor = tensor_plano.view(batch_size, self.height, self.width, 3)
        canvas = canvas_tensor[0].detach().cpu().numpy()
        
        np.random.seed(seed)
        t_fase = (seed % 1000) * 0.01
        matriz_generada = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        for i in range(self.height):
            y_norm = (i / self.height) * np.pi * 4.0
            for j in range(self.width):
                x_norm = (j / self.width) * np.pi * 4.0
                
                val_freq1 = np.sin(x_norm * 2.0 + t_fase) * np.cos(y_norm * 2.0 - t_fase)
                val_freq2 = np.sin(np.sqrt(x_norm**2 + y_norm**2) * 3.0 - t_fase * 2.0)
                val_interferencia = (val_freq1 + val_freq2) * 0.5
                
                pixel_neural = canvas[i, j]
                lambda_base = 520.0 + (val_interferencia * 140.0) + (pixel_neural[0] * 90.0)
                fr, fg, fb = self.fisica_longitud_onda_a_rgb(lambda_base)
                
                r_final = 0.5 * (pixel_neural[0] + 1.0) * 0.3 + fr * 0.7
                g_final = 0.5 * (pixel_neural[1] + 1.0) * 0.3 + fg * 0.7
                b_final = 0.5 * (pixel_neural[2] + 1.0) * 0.3 + fb * 0.7
                
                matriz_generada[i, j] = np.clip([r_final, g_final, b_final], 0.0, 1.0)
                
        return matriz_generada

# ==============================================================================
# 5. ARQUITECTURA MULTIMODAL UNIFICADA (MLPD + Q-MLPD + MOTOR GEOMÉTRICO)
# ==============================================================================

class MembranaLatentePermeable(nn.Module):
    def __init__(self, dz=32):
        super(MembranaLatentePermeable, self).__init__()
        self.umbral_permeabilidad = nn.Parameter(torch.ones(1, dz) * 0.5)
        self.factor_selectivo = nn.Parameter(torch.ones(1, dz) * 1.2)

    def forward(self, z_crudo, entropia_seq):
        portillo = torch.sigmoid(z_crudo * self.factor_selectivo - self.umbral_permeabilidad)
        z_filtrado = z_crudo * portillo * (1.0 + torch.tanh(entropia_seq.unsqueeze(1)))
        return z_filtrado

class OraculoCuanticoGenomico(nn.Module):
    """Módulo Q-MLPD: Simulación cuántica y de gravedad espacial[cite: 6]."""
    def __init__(self, dz=32, qubits_virtuales=8):
        super(OraculoCuanticoGenomico, self).__init__()
        self.dz = dz
        self.qubits = qubits_virtuales
        self.matriz_hadamard_simulada = nn.Parameter(torch.randn(qubits_virtuales, dz) * 0.1)
        self.fase_gravitacional = nn.Parameter(torch.tensor([9.80665]))

    def forward(self, Z_filtrado, factor_g=1.0):
        psi = torch.matmul(Z_filtrado, self.matriz_hadamard_simulada.T)
        interferencia = torch.cos(psi * (self.fase_gravitacional * factor_g))
        estado_colapsado = torch.sum(interferencia, dim=-1, keepdim=True)
        Z_cuantico = Z_filtrado * torch.sigmoid(estado_colapsado)
        return Z_cuantico

class EncodersMultimodalesUnificados(nn.Module):
    def __init__(self, p_dna=16, p_spatial=4, p_strat=4, d1=16, d2=16, d3=16, dz=32, width=256, height=256):
        super(EncodersMultimodalesUnificados, self).__init__()
        
        self.encoder_dna = nn.Sequential(nn.Linear(p_dna, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, d1))
        self.encoder_spatial = nn.Sequential(nn.Linear(p_spatial, 16), nn.LayerNorm(16), nn.GELU(), nn.Linear(16, d2))
        self.encoder_strat = nn.Sequential(nn.Linear(p_strat, 16), nn.LayerNorm(16), nn.GELU(), nn.Linear(16, d3))
        
        self.W1 = nn.Parameter(torch.randn(dz, d1) * 0.02)
        self.W2 = nn.Parameter(torch.randn(dz, d2) * 0.02)
        self.W3 = nn.Parameter(torch.randn(dz, d3) * 0.02)
        self.bias_z = nn.Parameter(torch.zeros(dz))
        
        self.activacion_z = nn.GELU()
        self.membrana_latente = MembranaLatentePermeable(dz=dz)
        self.oraculo_cuantico = OraculoCuanticoGenomico(dz=dz)
        self.factor_lambda_mod = FactorLambda(num_features=10)
        
        # Motor de Geometría Pura integrado para renderizar el fenotipo latente
        self.motor_geometrico = GeneradorGeometriaPuraMatematica(width=width, height=height, dz=dz)
        
        self.predictor = nn.Sequential(
            nn.Linear(dz + 4 + 10 + 25, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x_dna, x_spatial, x_strat, x_logicas, x_avanzados, x_matriz, entropia, factor_g=1.0, mask=None):
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
        
        Z_bruto = self.activacion_z(term1 + term2 + term3 + self.bias_z)
        Z_filtrado = self.membrana_latente(Z_bruto, entropia)
        Z = self.oraculo_cuantico(Z_filtrado, factor_g=factor_g)
        
        features_lambda_aplicados = self.factor_lambda_mod(x_avanzados)
        features_completos = torch.cat([Z, x_logicas, features_lambda_aplicados, x_matriz], dim=1)
        y_hat = self.predictor(features_completos)
        
        return y_hat, Z

def simular_y_graficar_espacio_cuantico(Z_tensor):
    oraculo = OraculoCuanticoGenomico(dz=Z_tensor.shape[1])
    z_tierra = oraculo(Z_tensor, factor_g=1.0).detach().numpy()
    z_lunar = oraculo(Z_tensor, factor_g=0.165).detach().numpy()
    z_cero = oraculo(Z_tensor, factor_g=0.001).detach().numpy()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z_tierra[:, 0], z_tierra[:, 1], z_tierra[:, 2], c='blue', label='Gravedad Terrestre (1.0g)', alpha=0.7)
    ax.scatter(z_lunar[:, 0], z_lunar[:, 1], z_lunar[:, 2], c='orange', label='Gravedad Lunar (0.165g)', alpha=0.7)
    ax.scatter(z_cero[:, 0], z_cero[:, 1], z_cero[:, 2], c='red', label='Microgravedad (0.001g)', alpha=0.7)

    ax.set_title("Deformación Topológica del Espacio Latente Z (Q-MLPD)")
    ax.set_xlabel("Eje Latente Q1")
    ax.set_ylabel("Eje Latente Q2")
    ax.set_zlabel("Eje Latente Q3")
    ax.legend()
    
    plt.savefig("reporte_cuantico_genomico.png", dpi=300)
    plt.close()
    print("[📈] Gráfica 3D guardada como 'reporte_cuantico_genomico.png'.")

# ==============================================================================
# 6. PIPELINE DE EJECUCIÓN PRINCIPAL INTEGRADO
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*78)
    print("PIPELINE HÍBRIDO  UNIFICADO V3.0.0: MLPD + Q-MLPD + MOTOR GEOMÉTRICO")
    print("="*78)

    diccionario_secuencias = leer_archivo_fasta(ARCHIVO_FASTA)
    targets_dict = cargar_targets_experimentales(ARCHIVO_TARGETS)

    ids_validos = []
    t_dna, t_spatial, t_strat, t_logicas, t_avanzados, t_matriz, t_entropia, t_targets = [], [], [], [], [], [], [], []

    print("\n--- [1/3] EXTRACCIÓN MODAL Y ANÁLISIS TOPOLÓGICO GENÓMICO ---")
    for id_seq, seq in diccionario_secuencias.items():
        seq_limpia = limpiar_secuencia(seq)
        if len(seq_limpia) < 5:
            continue
        
        longitud = len(seq_limpia)
        freqs = [seq_limpia.count(b) / longitud for b in "ATGC"]
        hidros = [TABLA_QUIMICA_ADN[b]['hidropatia'] for b in seq_limpia]
        pesos = [TABLA_QUIMICA_ADN[b]['peso_molecular'] for b in seq_limpia]
        
        ddna = freqs + [np.mean(hidros), np.std(hidros), np.mean(pesos), np.std(pesos)] + [0.0] * 8
        ddna = ddna[:16]

        ciclos = analizar_ciclos_secuencia(seq_limpia, k=3)
        G = nx.DiGraph()
        for i in range(len(seq_limpia) - 3):
            G.add_edge(seq_limpia[i:i+3], seq_limpia[i+1:i+1+3])
        dspatial = [float(ciclos), float(G.number_of_nodes()), float(G.number_of_edges()), np.linalg.norm(hidros)]

        derivada = np.gradient(hidros) if len(hidros) > 1 else np.array([0.0])
        derivada_rms = float(np.sqrt(np.mean(derivada**2)))
        integral_area = float(trapezoid(hidros, dx=1.0))
        log_entropia = float(-sum(f * np.log2(f) for f in freqs if f > 0))
        dstrat = [derivada_rms, integral_area, log_entropia, float(longitud)]

        purinas = (seq_limpia.count("A") + seq_limpia.count("G")) / longitud
        gc_ratio = (seq_limpia.count("G") + seq_limpia.count("C")) / longitud
        logicas = evaluar_logica_difusa_genomica(
            purinas, gc_ratio, log_entropia, float(longitud), 
            seq_limpia.startswith("AT"), "GGG" in seq_limpia or "CCC" in seq_limpia
        ).numpy()

        adv, matriz_comp = calcular_perfil_matematico_avanzado(seq_limpia)
        target = targets_dict.get(id_seq, 0.85)
        
        ids_validos.append(id_seq)
        t_dna.append(ddna)
        t_spatial.append(dspatial)
        t_strat.append(dstrat)
        t_logicas.append(logicas)
        t_avanzados.append(adv)
        t_matriz.append(matriz_comp.flatten())
        t_entropia.append(log_entropia)
        t_targets.append([target])
        print(f"  [✔] ID: {id_seq} | Longitud: {longitud} | Entropía: {log_entropia:.4f}")

    X_dna = torch.tensor(np.array(t_dna), dtype=torch.float32)
    X_spatial = torch.tensor(np.array(t_spatial), dtype=torch.float32)
    X_strat = torch.tensor(np.array(t_strat), dtype=torch.float32)
    X_logicas = torch.tensor(np.array(t_logicas), dtype=torch.float32)
    X_avanzados = torch.tensor(np.array(t_avanzados), dtype=torch.float32)
    X_matriz = torch.tensor(np.array(t_matriz), dtype=torch.float32)
    X_entropia = torch.tensor(np.array(t_entropia), dtype=torch.float32)
    Y = torch.tensor(np.array(t_targets), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- [2/3] INICIALIZANDO MODELO UNIFICADO EN HARDWARE: {device} ---")
    
    modelo = EncodersMultimodalesUnificados().to(device)
    criterio = nn.MSELoss()
    optimizador = optim.AdamW(modelo.parameters(), lr=0.002, weight_decay=1e-4)

    X_dna_d = X_dna.to(device)
    X_spatial_d = X_spatial.to(device)
    X_strat_d = X_strat.to(device)
    X_logicas_d = X_logicas.to(device)
    X_avanzados_d = X_avanzados.to(device)
    X_matriz_d = X_matriz.to(device)
    X_entropia_d = X_entropia.to(device)
    Y_d = Y.to(device)

    epochs = 1000
    for epoch in range(epochs):
        modelo.train()
        optimizador.zero_grad()
        predicciones, _ = modelo(
            X_dna_d, X_spatial_d, X_strat_d, X_logicas_d, 
            X_avanzados_d, X_matriz_d, X_entropia_d, factor_g=1.0
        )
        loss = criterio(predicciones, Y_d)
        loss.backward()
        optimizador.step()
        if (epoch + 1) % 500 == 0:
            print(f"Época [{epoch+1}/{epochs}] | MSE Loss: {loss.item():.7f}")

    print("\n--- [3/3] INFERENCIA CUÁNTICA Y RENDERIZADO GEOMÉTRICO EX NIHILO ---")
    modelo.eval()
    with torch.no_grad():
        y_final, Z_representacion = modelo(
            X_dna_d, X_spatial_d, X_strat_d, X_logicas_d, 
            X_avanzados_d, X_matriz_d, X_entropia_d, factor_g=1.0
        )
        
        for i, id_s in enumerate(ids_validos):
            print(f"  - ID: {id_s} | Predicción: {y_final[i].item():.6f}")

        # Sintetizar visualmente el fenotipo geométrico a partir del primer vector latente Z
        z_sample = Z_representacion[0:1]
        matriz_fenotipo = modelo.motor_geometrico.forward_generar_geometria(z_sample, seed=42)
        imagen_fenotipo = Image.fromarray((matriz_fenotipo * 255).astype(np.uint8), 'RGB')
        imagen_fenotipo.save("fenotipo_genomico_sintetizado.png")
        print("[🎨] Fenotipo geométrico visual generado y guardado como 'fenotipo_genomico_sintetizado.png'.")

        simular_y_graficar_espacio_cuantico(Z_representacion.cpu())

    print("\n" + "="*78)
    print("EJECUCIÓN UNIFICADA COMPLETADA EXITOSAMENTE")
    print("="*78)