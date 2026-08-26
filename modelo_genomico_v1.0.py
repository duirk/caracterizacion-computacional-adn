
# ==============================================================================
# PASO DE SIMULACIÓN A DATOS REALES
#
# 1. DATOS DE ENTRADA:
#    Coloca en esta misma carpeta un archivo FASTA real llamado:
#        'genoma_muestra.fasta'
#    El archivo debe contener las secuencias que se utilizarán como entradas
#    del modelo.
#
# 2. ETIQUETAS EXPERIMENTALES (TARGET):
#    Actualmente Y_tensor contiene valores simulados (0.92).
#    Para utilizar datos reales, sustituye estos valores por mediciones
#    experimentales obtenidas en laboratorio.
#
#    Debe existir una etiqueta experimental correspondiente a cada secuencia
#    del FASTA. Por ejemplo, la etiqueta podría representar afinidad de unión,
#    actividad/inhibición, expresión, viabilidad celular u otra variable
#    cuantitativa definida experimentalmente.
#
#    IMPORTANTE: las etiquetas deben estar correctamente asociadas con el ID
#    de cada secuencia y utilizar una escala bien definida.
#
# 3. DATASET Y VALIDACIÓN:
#    Para obtener un modelo generalizable se recomienda utilizar muchas
#    secuencias reales y dividir los datos en conjuntos independientes de
#    entrenamiento, validación y prueba.
#
# 4. VALIDACIÓN EXPERIMENTAL:
#    Las predicciones computacionales deben evaluarse frente a resultados
#    experimentales independientes (wet lab). El modelo no debe considerarse
#    validado únicamente porque la pérdida de entrenamiento sea baja.
#
# 5. REPRODUCIBILIDAD:
#    Conserva los metadatos de las secuencias, las mediciones experimentales,
#    las unidades y las condiciones experimentales para poder auditar la
#    correspondencia entre FASTA, etiquetas y predicciones.
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
from scipy.integrate import trapezoid
import os

# ==========================================
# 1. TABLA DE INFORMACIÓN QUÍMICA Y BIOFÍSICA
# ==========================================
TABLA_QUIMICA_ADN = {
    'A': {'peso_molecular': 135.13, 'hidropatia': -1.9, 'puentes_h': 2},
    'T': {'peso_molecular': 126.11, 'hidropatia': -0.7, 'puentes_h': 2},
    'G': {'peso_molecular': 151.13, 'hidropatia': -3.5, 'puentes_h': 3},
    'C': {'peso_molecular': 111.10, 'hidropatia': -3.5, 'puentes_h': 3}
}

# ==========================================
# 2. PARSER DE ARCHIVOS REALES (FORMATO FASTA)
# ==========================================
def leer_archivo_fasta(ruta_archivo):
    """Lee un archivo FASTA real y extrae las secuencias genómicas omitiendo cabeceras."""
    secuencias = {}
    if not os.path.exists(ruta_archivo):
        print(f"[!] No se encontró el archivo {ruta_archivo}. Usando secuencia por defecto.")
        return {"Secuencia_Defecto": "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGA"}
    
    current_header = ""
    with open(ruta_archivo, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_header = line[1:]
                secuencias[current_header] = ""
            elif current_header:
                secuencias[current_header] += line.upper()
                
    return secuencias

# ==========================================
# 3. CÁLCULO MATEMÁTICO Y TOPOLÓGICO AVANZADO
# ==========================================
def calcular_perfil_matematico(secuencia):
    """Calcula derivadas, integrales discretas y entropía estocástica."""
    secuencia = ''.join([b for b in secuencia.upper() if b in "ATGC"])
    if len(secuencia) < 2:
        return 0.0, 0.0, 0.0

    valores_hidropatia = [TABLA_QUIMICA_ADN[base]['hidropatia'] for base in secuencia]

    # Derivada discreta (cambio de hidropatía entre bases consecutivas)
    derivada = np.gradient(valores_hidropatia)
    derivada_rms = float(np.sqrt(np.mean(derivada**2)))

    # Integral numérica directa (área bajo la curva hidropática)
    integral_area = float(trapezoid(valores_hidropatia, dx=1.0))

    # Logaritmo de entropía de Shannon (complejidad estocástica)
    frecuencias = {b: secuencia.count(b) / len(secuencia) for b in "ATGC"}
    log_entropia = -sum(f * np.log2(f) for f in frecuencias.values() if f > 0)

    return derivada_rms, float(integral_area), float(log_entropia)

# ==========================================
# 4. TEOREMA DE CICLOS EN GRAFOS (K-MERS)
# ==========================================
def analizar_ciclos_secuencia(secuencia):
    """Construye un grafo dirigido para detectar bucles y ciclos estructurales."""
    G = nx.DiGraph()
    k = 3  # Codones de tamaño 3
    
    for i in range(len(secuencia) - k):
        kmer_origen = secuencia[i:i+k]
        kmer_destino = secuencia[i+1:i+1+k]
        G.add_edge(kmer_origen, kmer_destino)
        
    try:
        ciclos = nx.simple_cycles(G)
        num_ciclos = sum(1 for _ in ciclos)
    except Exception:
        num_ciclos = 0
        
    return num_ciclos

# ==========================================
# 5. RED NEURONAL PROBABILÍSTICA EN CUDA
# ==========================================
class RedGenomicaAvanzada(nn.Module):
    def __init__(self):
        super(RedGenomicaAvanzada, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2) # Mitiga sobreajuste
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        if x.size(0) > 1: # BatchNorm requiere batch > 1
            out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.sigmoid(out)

def entrenar_modelo_cuda(features_tensor, target_tensor):
    """Entrena la red neuronal acelerada por hardware CUDA."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Hardware seleccionado para IA: {device}")
    
    modelo = RedGenomicaAvanzada().to(device)
    criterio = nn.MSELoss()
    optimizador = optim.Adam(modelo.parameters(), lr=0.005)
    
    X = features_tensor.to(device)
    Y = target_tensor.to(device)
    
    max_epochs = 1500
    for epoch in range(max_epochs):
        modelo.train()
        optimizador.zero_grad()
        prediccion = modelo(X)
        loss = criterio(prediccion, Y)
        loss.backward()
        optimizador.step()
        
        if loss.item() < 0.00005:
            print(f"[+] Convergencia alcanzada en Época {epoch} | Pérdida: {loss.item():.7f}")
            break
            
    return modelo

# ==========================================
# 6. PIPELINE DE EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # Simulación de lectura (puedes cambiar 'genoma.fasta' por un archivo real)
    archivo_fasta = "genoma_muestra.fasta"
    diccionario_secuencias = leer_archivo_fasta(archivo_fasta)
    
    dataset_features = []
    
    print("\n--- PROCESANDO SECUENCIAS ---")
    for id_seq, seq in diccionario_secuencias.items():
        der, integ, log_ent = calcular_perfil_matematico(seq)
        ciclos = analizar_ciclos_secuencia(seq)
        longitud = len(seq)
        
        print(f"ID: {id_seq} | Longitud: {longitud} | Derivada: {der:.2f} | Ciclos: {ciclos}")
        dataset_features.append([der, integ, log_ent, float(ciclos), float(longitud)])

    # Conversión a tensores para PyTorch / CUDA
    X_tensor = torch.tensor(dataset_features, dtype=torch.float32)
    # Objetivo simulado de eficacia (valores de referencia para entrenamiento)
    Y_tensor = torch.tensor([[0.92] for _ in dataset_features], dtype=torch.float32)
    
    print("\n--- INICIANDO ENTRENAMIENTO EN GPU ---")
    modelo_final = entrenar_modelo_cuda(X_tensor, Y_tensor)
    
    # Inferencia final de prueba con el modelo entrenado
    modelo_final.eval()
    with torch.no_grad():
        device = next(modelo_final.parameters()).device
        resultado_inferencia = modelo_final(X_tensor.to(device))
        print(f"\n[✔] Predicción de Eficacia Terapéutica (Inferencia GPU): {resultado_inferencia.cpu().numpy().flatten()}")