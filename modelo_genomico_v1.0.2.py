# ==============================================================================
# PIPELINE HÍBRIDO AVANZADO: ARQUITECTURA MULTIMODAL CON ESPACIO LATENTE Z
# Basado en la especificación formal de Nadal Ferrà (2026)
# ==============================================================================

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
from scipy.integrate import trapezoid

# ==============================================================================
# 0. CONFIGURACIÓN Y BIOFÍSICA BÁSICA
# ==============================================================================

ARCHIVO_FASTA = "genoma_muestra.fasta"
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

TABLA_QUIMICA_ADN = {
    'A': {'peso_molecular': 135.13, 'hidropatia': -1.9, 'puentes_h': 2},
    'T': {'peso_molecular': 126.11, 'hidropatia': -0.7, 'puentes_h': 2},
    'G': {'peso_molecular': 151.13, 'hidropatia': -3.5, 'puentes_h': 3},
    'C': {'peso_molecular': 111.10, 'hidropatia': -3.5, 'puentes_h': 3}
}

# ==============================================================================
# 1. PARSER FASTA Y EXTRACCIÓN DE MODALIDADES HETEROGÉNEAS
# ==============================================================================

def leer_archivo_fasta(ruta_archivo):
    secuencias = {}
    if not os.path.exists(ruta_archivo):
        print(f"[!] No se encontró {ruta_archivo}. Usando secuencia por defecto.")
        return {"Secuencia_Defecto": "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGA"}
    
    current_header = ""
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_header = line[1:].strip()
                secuencias[current_header] = ""
            elif current_header:
                secuencias[current_header] += line.upper()
    return secuencias

def extraer_modalidades_crudas(secuencia):
    """
    Genera los vectores de entrada crudos para las 3 modalidades:
    - DDNA (Genómico): Composición y propiedades base a base (longitud fija o rellenada).
    - Dspatial (Geométrico): Descriptores espectrales, grafos y ciclos de k-mers.
    - Dstrat (Estratigráfico): Integrales funcionales y perfiles de profundidad/hidropatía.
    """
    secuencia = ''.join([b for b in secuencia.upper() if b in "ATGC"])
    if len(secuencia) < 5:
        secuencia = "ATGCGATCGATC"

    # 1. Modalidad Genómica (DDNA) -> Vector de frecuencias y propiedades físicas normalizadas
    longitud = len(secuencia)
    freqs = [secuencia.count(b) / longitud for b in "ATGC"]
    hidros = [TABLA_QUIMICA_ADN[b]['hidropatia'] for b in secuencia]
    pesos = [TABLA_QUIMICA_ADN[b]['peso_molecular'] for b in secuencia]
    
    # DDNA dimensión fija de 16 elementos representativos de la secuencia
    ddna = freqs + [np.mean(hidros), np.std(hidros), np.mean(pesos), np.std(pesos)] + [0.0] * 8
    ddna = ddna[:16]

    # 2. Modalidad Geométrica (Dspatial) -> Grafo de k-mers y autovalores simulados
    G = nx.DiGraph()
    k = 3
    for i in range(len(secuencia) - k):
        G.add_edge(secuencia[i:i+k], secuencia[i+1:i+1+k])
    try:
        num_ciclos = sum(1 for _ in nx.simple_cycles(G))
    except:
        num_ciclos = 0
    
    dspatial = [float(num_ciclos), float(G.number_of_nodes()), float(G.number_of_edges()), np.linalg.norm(hidros) if len(hidros)>0 else 0.0]

    # 3. Modalidad Estratigráfica (Dstrat) -> Derivadas, integrales y entropía de Shannon
    derivada = np.gradient(hidros) if len(hidros) > 1 else np.array([0.0])
    derivada_rms = float(np.sqrt(np.mean(derivada**2)))
    integral_area = float(trapz_safe(hidro_vals=hidros))
    log_entropia = -sum(f * np.log2(f) for f in freqs if f > 0)
    
    dstrat = [derivada_rms, integral_area, log_entropia, float(longitud)]

    return np.array(ddna, dtype=np.float32), np.array(dspatial, dtype=np.float32), np.array(dstrat, dtype=np.float32)

def trapz_safe(hidro_vals):
    if len(hidro_vals) < 2:
        return 0.0
    return float(trapezoid(hidro_vals, dx=1.0))

# ==============================================================================
# 2. ARQUITECTURA MULTIMODAL CON ESPACIO LATENTE Z (ENCODERS + FUSIÓN)
# ==============================================================================

class EncodersMultimodalesYZ(nn.Module):
    def __init__(self, p_dna=16, p_spatial=4, p_strat=4, d1=16, d2=16, d3=16, dz=32):
        super(EncodersMultimodalesYZ, self).__init__()
        
        # Encoder Genómico (EDNA) - Convoluciones 1D simuladas / MLP denso equivalente
        self.encoder_dna = nn.Sequential(
            nn.Linear(p_dna, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, d1)
        )
        
        # Encoder Geométrico (Espatial)
        self.encoder_spatial = nn.Sequential(
            nn.Linear(p_spatial, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, d2)
        )
        
        # Encoder Estratigráfico (Estrat)
        self.encoder_strat = nn.Sequential(
            nn.Linear(p_strat, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, d3)
        )
        
        # Matrices de proyección global para la Fusión Tardía hacia Z
        # W1, W2, W3 proyectan cada modalidad al espacio latente compartido Z de dimensión dz
        self.W1 = nn.Parameter(torch.randn(dz, d1) * 0.02)
        self.W2 = nn.Parameter(torch.randn(dz, d2) * 0.02)
        self.W3 = nn.Parameter(torch.randn(dz, d3) * 0.02)
        self.bias_z = nn.Parameter(torch.zeros(dz))
        
        self.activacion_z = nn.GELU()
        
        # Módulo predictor posterior g_theta a partir de Z
        self.predictor = nn.Sequential(
            nn.Linear(dz, 16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x_dna, x_spatial, x_strat, mask=None):
        """
        Calcula la representación en el espacio latente Z y la predicción final.
        Soporta robustez ante ausencia parcial de datos mediante máscaras opcionales.
        """
        # Si se simula ausencia de datos (enmascaramiento a ceros)
        if mask is not None:
            x_dna = x_dna * mask[:, 0:1]
            x_spatial = x_spatial * mask[:, 1:2]
            x_strat = x_strat * mask[:, 2:3]

        # 1. Transformación independiente por Encoders
        e_dna = self.encoder_dna(x_dna)         # [Batch, d1]
        e_spatial = self.encoder_spatial(x_spatial) # [Batch, d2]
        e_strat = self.encoder_strat(x_strat)     # [Batch, d3]

        # 2. Fusión Tardía Proyectada en el Espacio Latente Z:
        # Z = GELU( E_dna @ W1^T + E_spatial @ W2^T + E_strat @ W3^T + b )
        term1 = torch.matmul(e_dna, self.W1.T)
        term2 = torch.matmul(e_spatial, self.W2.T)
        term3 = torch.matmul(e_strat, self.W3.T)
        
        Z = self.activacion_z(term1 + term2 + term3 + self.bias_z) # [Batch, dz]

        # 3. Predicción posterior final (y_hat)
        y_hat = self.predictor(Z)
        
        return y_hat, Z

# ==============================================================================
# 3. PIPELINE DE ENTRENAMIENTO Y VALIDACIÓN EXPERIMENTAL
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*78)
    print("PIPELINE EXPERIMENTAL: ARQUITECTURA MULTIMODAL CON ESPACIO LATENTE Z")
    print("="*78)

    # 1. Cargar datos FASTA
    diccionario_secuencias = leer_archivo_fasta(ARCHIVO_FASTA)
    
    ids = []
    tensor_dna, tensor_spatial, tensor_strat = [], [], []

    print("\n--- [1/4] EXRAYENDO MODALIDADES HETEROGÉNEAS ---")
    for id_seq, seq in diccionario_secuencias.items():
        ddna, dspatial, dstrat = extraer_modalidades_crudas(seq)
        ids.append(id_seq)
        tensor_dna.append(ddna)
        tensor_spatial.append(dspatial)
        tensor_strat.append(dstrat)
        print(f"ID: {id_seq} | Longitud: {len(seq)} procesada correctamente.")

    X_dna = torch.tensor(np.array(tensor_dna), dtype=torch.float32)
    X_spatial = torch.tensor(np.array(tensor_spatial), dtype=torch.float32)
    X_strat = torch.tensor(np.array(tensor_strat), dtype=torch.float32)
    
    # Target experimental simulado (o real si se conecta a un CSV)
    Y = torch.tensor([[0.85] for _ in ids], dtype=torch.float32)

    # 2. Inicializar el modelo multimodal
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- [2/4] INICIALIZANDO MODELO EN HARDWARE: {device} ---")
    
    modelo = EncodersMultimodalesYZ().to(device)
    criterio = nn.MSELoss()
    optimizador = optim.AdamW(modelo.parameters(), lr=0.003, weight_decay=1e-4)

    X_dna_d = X_dna.to(device)
    X_spatial_d = X_spatial.to(device)
    X_strat_d = X_strat.to(device)
    Y_d = Y.to(device)

    # 3. Entrenamiento optimizado
    print("\n--- [3/4] ENTRENAMIENTO DEL ESPACIO LATENTE Z ---")
    epochs = 1000
    for epoch in range(epochs):
        modelo.train()
        optimizador.zero_grad()
        
        predicciones, Z_latente = modelo(X_dna_d, X_spatial_d, X_strat_d)
        loss = criterio(predicciones, Y_d)
        
        loss.backward()
        optimizador.step()

        if (epoch + 1) % 250 == 0 or loss.item() < 0.00001:
            print(f"Época [{epoch+1}/{epochs}] | MSE Loss: {loss.item():.7f}")
        if loss.item() < 0.00001:
            print(f"[+] Convergencia lograda en época {epoch+1}")
            break

    # 4. Evaluación de Inferencia y Robustez (Prueba de Ausencia Parcial de Datos)
    print("\n--- [4/4] INFERENCIA Y TEST DE ROBUSTEZ ANTE DATOS FALTANTES ---")
    modelo.eval()
    with torch.no_grad():
        # Test normal
        y_final, Z_representacion = modelo(X_dna_d, X_spatial_d, X_strat_d)
        
        print(f"\n[✔] Dimensión del Espacio Latente Z obtenido: {Z_representacion.shape}")
        for i, id_s in enumerate(ids):
            print(f"  - ID: {id_s} | Predicción Base: {y_final[i].item():.4f}")

        # Test de Robustez: Enmascarando la modalidad estratigráfica a cero (simulando datos perdidos)
        mask_ausencia = torch.ones((len(ids), 3), device=device)
        mask_ausencia[:, 2] = 0.0 # Apagamos modalidad estratigráfica artificialmente
        
        y_robusto, _ = modelo(X_dna_d, X_spatial_d, X_strat_d, mask=mask_ausencia)
        print("\n[🛡] PRUEBA DE ROBUSTEZ (Ausencia de Datos Estratigráficos):")
        for i, id_s in enumerate(ids):
            print(f"  - ID: {id_s} | Predicción con datos faltantes: {y_robusto[i].item():.4f}")

    print("\n" + "="*78)
    print("EJECUCIÓN COMPLETADA EXITOSAMENTE")
    print("="*78)