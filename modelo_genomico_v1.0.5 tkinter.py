# ==============================================================================
# INTERFAZ GRÁFICA AVANZADA V3.0.0: MLPD + Q-MLPD + MOTOR GEOMÉTRICO EX NIHILO
# ==============================================================================

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import torch
import os
import importlib.util
import sys
import networkx as nx  # <-- Añade esta línea aquí
from PIL import Image
# Carga dinámica segura del módulo unificado V3.0.0
nombre_archivo = "modelo_genomico_v1.0.5.py"
modelo_bio = None
...
# Carga dinámica segura del módulo unificado V3.0.0
nombre_archivo = "modelo_genomico_v1.0.5.py"
modelo_bio = None

if os.path.exists(nombre_archivo):
    try:
        spec = importlib.util.spec_from_file_location("modelo_bio", nombre_archivo)
        modelo_bio = importlib.util.module_from_spec(spec)
        sys.modules["modelo_bio"] = modelo_bio
        spec.loader.exec_module(modelo_bio)
    except Exception as e:
        print(f"Error cargando el módulo: {e}")

class AppModeloGenomicoEpica:
    def __init__(self, root):
        self.root = root
        self.root.title("🧬 Pipeline Genómico Unificado V3.0.0 - MLPD, Q-MLPD & Geometría Ex Nihilo")
        self.root.geometry("1050x800")
        self.root.configure(bg="#090d16")

        self.estilo = ttk.Style()
        self.estilo.theme_use("clam")
        
        self.COLOR_BG = "#090d16"
        self.COLOR_ACCENT = "#0ea5e9"
        self.COLOR_TEXTO = "#f8fafc"

        self.configurar_estilos()
        self.crear_widgets()

    def configurar_estilos(self):
        self.estilo.configure("TNotebook", background=self.COLOR_BG, borderwidth=0)
        self.estilo.configure("TNotebook.Tab", background="#1e293b", foreground="#94a3b8", padding=[15, 8], font=("Segoe UI", 10, "bold"))
        self.estilo.map("TNotebook.Tab", background=[("selected", self.COLOR_ACCENT)], foreground=[("selected", "#ffffff")])
        self.estilo.configure("TFrame", background=self.COLOR_BG)
        self.estilo.configure("TLabel", background=self.COLOR_BG, foreground=self.COLOR_TEXTO, font=("Segoe UI", 10))

    def crear_widgets(self):
        header_frame = tk.Frame(self.root, bg="#0f172a", pady=15, padx=20)
        header_frame.pack(fill=tk.X)

        tk.Label(
            header_frame, 
            text="🧬 PIPELINE HÍBRIDO DOCTORAL UNIFICADO V3.0.0", 
            font=("Segoe UI", 14, "bold"), 
            fg="#38bdf8", 
            bg="#0f172a"
        ).pack(anchor=tk.W)

        tk.Label(
            header_frame, 
            text="Integración de Membranas Latentes (MLPD), Oráculo Cuántico (Q-MLPD) y Motor Geométrico Ex Nihilo", 
            font=("Segoe UI", 9), 
            fg="#94a3b8", 
            bg="#0f172a"
        ).pack(anchor=tk.W, pady=(2, 0))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Pestaña 1: Análisis Biofísico
        self.tab_analisis = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_analisis, text="  🔬 Análisis Biofísico & Funciones  ")
        self.construir_pestana_analisis()

        # Pestaña 2: Pipeline PyTorch / Unificado
        self.tab_pipeline = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_pipeline, text="  ⚡ Pipeline PyTorch / Q-MLPD & Visual  ")
        self.construir_pestana_pipeline()

        # Barra de estado inferior
        estado_frame = tk.Frame(self.root, bg="#020617", pady=5, padx=10)
        estado_frame.pack(fill=tk.X, side=tk.BOTTOM)

        estado_txt = "Estado: Listo | Módulo 'modelo_genomico_v1.0.5.py' vinculado con éxito." if modelo_bio else "Estado: [!] ADVERTENCIA - No se pudo vincular 'modelo_genomico_v1.0.5.py'"
        self.lbl_estado = tk.Label(
            estado_frame, 
            text=estado_txt, 
            fg="#34d399" if modelo_bio else "#f87171", 
            bg="#020617",
            font=("Segoe UI", 9)
        )
        self.lbl_estado.pack(anchor=tk.W)

    def construir_pestana_analisis(self):
        container = ttk.Frame(self.tab_analisis)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(container, text="Introduce la Secuencia de ADN (A, T, G, C):", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=5)

        self.txt_secuencia = tk.Text(container, height=4, font=("Consolas", 10), bg="#0f172a", fg="#38bdf8", insertbackground="white", bd=1, relief="solid")
        self.txt_secuencia.pack(fill=tk.X, pady=5)
        self.txt_secuencia.insert(tk.END, "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGA")

        tk.Button(
            container, 
            text="🚀 Ejecutar Extracción y Funciones Matemáticas", 
            command=self.ejecutar_analisis_secuencia,
            bg="#0284c7", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=10, pady=6, cursor="hand2"
        ).pack(anchor=tk.W, pady=10)

        ttk.Label(container, text="Consola de Resultados Matemáticos:", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=5)

        self.txt_salida_analisis = scrolledtext.ScrolledText(
            container, height=16, font=("Consolas", 9), 
            bg="#020617", fg="#4ade80", insertbackground="white",
            bd=1, relief="solid"
        )
        self.txt_salida_analisis.pack(fill=tk.BOTH, expand=True, pady=5)

    def construir_pestana_pipeline(self):
        container = ttk.Frame(self.tab_pipeline)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(
            container, 
            text="Gestión de Datos FASTA, Entrenamiento Unificado, Co-Simulación Espacial y Renderizado Visual.", 
            font=("Segoe UI", 10)
        ).pack(anchor=tk.W, pady=5)

        # Panel de controles para gravedad simulada y semillas
        controles_frame = ttk.Frame(container)
        controles_frame.pack(fill=tk.X, pady=5)

        ttk.Label(controles_frame, text="Factor Gravitacional (factor_g):").pack(side=tk.LEFT, padx=(0, 10))
        self.combo_gravedad = ttk.Combobox(controles_frame, values=["1.0 (Tierra)", "0.165 (Lunar)", "0.001 (Microgravedad)"], width=22, state="readonly")
        self.combo_gravedad.pack(side=tk.LEFT, padx=5)
        self.combo_gravedad.set("1.0 (Tierra)")

        tk.Button(
            container, 
            text="⚡ Inicializar Pipeline Unificado, Entrenar & Sintetizar Fenotipo Visual", 
            command=self.ejecutar_pipeline_completo,
            bg="#10b981", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=10, pady=6, cursor="hand2"
        ).pack(anchor=tk.W, pady=10)

        self.txt_salida_pipeline = scrolledtext.ScrolledText(
            container, height=18, font=("Consolas", 9), 
            bg="#020617", fg="#38bdf8", insertbackground="white",
            bd=1, relief="solid"
        )
        self.txt_salida_pipeline.pack(fill=tk.BOTH, expand=True, pady=5)

    def ejecutar_analisis_secuencia(self):
        if not modelo_bio:
            messagebox.showerror("Error Crítico", "El archivo 'modelo_genomico_v1.0.5.py' no se encuentra disponible.")
            return

        secuencia_bruta = self.txt_secuencia.get("1.0", tk.END).strip()
        secuencia_limpia = modelo_bio.limpiar_secuencia(secuencia_bruta)

        if len(secuencia_limpia) < 5:
            messagebox.showwarning("Aviso", "La secuencia debe contener al menos 5 bases válidas (A, T, G, C).")
            return

        self.txt_salida_analisis.delete("1.0", tk.END)
        self.txt_salida_analisis.insert(tk.END, f">> Secuencia Procesada: {secuencia_limpia}\n")
        self.txt_salida_analisis.insert(tk.END, f">> Longitud Total: {len(secuencia_limpia)} bases\n")
        self.txt_salida_analisis.insert(tk.END, "="*65 + "\n\n")

        try:
            ciclos = modelo_bio.analizar_ciclos_secuencia(secuencia_limpia, k=3)
            self.txt_salida_analisis.insert(tk.END, f"[✔] Ciclos Topológicos en Grafo (k=3): {ciclos}\n")

            vector_feat, matriz_comp = modelo_bio.calcular_perfil_matematico_avanzado(secuencia_limpia)
            self.txt_salida_analisis.insert(tk.END, f"[✔] Vector Features Matemáticos (10 dims):\n    {vector_feat}\n\n")
            self.txt_salida_analisis.insert(tk.END, f"[✔] Matriz de Transformación 5x5 Aplanada:\n    {matriz_comp.flatten()[:5]} ... [Total 25 dims]\n\n")

            longitud = len(secuencia_limpia)
            freqs = [secuencia_limpia.count(b) / longitud for b in "ATGC"]
            purinas = (secuencia_limpia.count("A") + secuencia_limpia.count("G")) / longitud
            gc_ratio = (secuencia_limpia.count("G") + secuencia_limpia.count("C")) / longitud
            log_entropia = float(-sum(f * np.log2(f) for f in freqs if f > 0))
            
            logicas_difusas = modelo_bio.evaluar_logica_difusa_genomica(
                purinas, gc_ratio, log_entropia, float(longitud), 
                secuencia_limpia.startswith("AT"), "GGG" in secuencia_limpia or "CCC" in secuencia_limpia
            ).numpy()

            self.txt_salida_analisis.insert(tk.END, f"[✔] Entropía de Shannon Calculada: {log_entropia:.5f}\n")
            self.txt_salida_analisis.insert(tk.END, f"[✔] Lógica Difusa Genómica (Receptor, Mutación, Resistencia, Eficacia):\n    {logicas_difusas}\n")
            
            self.lbl_estado.config(text="Estado: Análisis biofísico ejecutado correctamente.")
        except Exception as e:
            messagebox.showerror("Excepción de Ejecución", f"Ocurrió un error al procesar las funciones: {str(e)}")

    def ejecutar_pipeline_completo(self):
        if not modelo_bio:
            messagebox.showerror("Error Crítico", "No se puede ejecutar el pipeline sin el módulo base unificado.")
            return

        # Mapear selección de gravedad
        seleccion_g = self.combo_gravedad.get()
        if "Lunar" in seleccion_g:
            factor_g = 0.165
        elif "Microgravedad" in seleccion_g:
            factor_g = 0.001
        else:
            factor_g = 1.0

        self.txt_salida_pipeline.delete("1.0", tk.END)
        self.txt_salida_pipeline.insert(tk.END, f"[*] Iniciando Pipeline Unificado (factor_g={factor_g})...\n")
        self.root.update_idletasks()

        try:
            diccionario_secuencias = modelo_bio.leer_archivo_fasta(modelo_bio.ARCHIVO_FASTA)
            targets_dict = modelo_bio.cargar_targets_experimentales(modelo_bio.ARCHIVO_TARGETS)

            ids_validos = []
            t_dna, t_spatial, t_strat, t_logicas, t_avanzados, t_matriz, t_entropia, t_targets = [], [], [], [], [], [], [], []

            for id_seq, seq in diccionario_secuencias.items():
                seq_limpia = modelo_bio.limpiar_secuencia(seq)
                if len(seq_limpia) < 5:
                    continue
                
                longitud = len(seq_limpia)
                freqs = [seq_limpia.count(b) / longitud for b in "ATGC"]
                hidros = [modelo_bio.TABLA_QUIMICA_ADN[b]['hidropatia'] for b in seq_limpia]
                pesos = [modelo_bio.TABLA_QUIMICA_ADN[b]['peso_molecular'] for b in seq_limpia]
                
                ddna = freqs + [np.mean(hidros), np.std(hidros), np.mean(pesos), np.std(pesos)] + [0.0] * 8
                ddna = ddna[:16]

                ciclos = modelo_bio.analizar_ciclos_secuencia(seq_limpia, k=3)
                G = nx.DiGraph()
                for i in range(len(seq_limpia) - 3):
                    G.add_edge(seq_limpia[i:i+3], seq_limpia[i+1:i+1+3])
                dspatial = [float(ciclos), float(G.number_of_nodes()), float(G.number_of_edges()), np.linalg.norm(hidros)]

                derivada = np.gradient(hidros) if len(hidros) > 1 else np.array([0.0])
                derivada_rms = float(np.sqrt(np.mean(derivada**2)))
                integral_area = float(modelo_bio.trapezoid(hidros, dx=1.0))
                log_entropia = float(-sum(f * np.log2(f) for f in freqs if f > 0))
                dstrat = [derivada_rms, integral_area, log_entropia, float(longitud)]

                purinas = (seq_limpia.count("A") + seq_limpia.count("G")) / longitud
                gc_ratio = (seq_limpia.count("G") + seq_limpia.count("C")) / longitud
                logicas = modelo_bio.evaluar_logica_difusa_genomica(
                    purinas, gc_ratio, log_entropia, float(longitud), 
                    seq_limpia.startswith("AT"), "GGG" in seq_limpia or "CCC" in seq_limpia
                ).numpy()

                adv, matriz_comp = modelo_bio.calcular_perfil_matematico_avanzado(seq_limpia)
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

            if not ids_validos:
                self.txt_salida_pipeline.insert(tk.END, "[!] No se encontraron secuencias válidas en el FASTA.\n")
                return

            X_dna = torch.tensor(np.array(t_dna), dtype=torch.float32)
            X_spatial = torch.tensor(np.array(t_spatial), dtype=torch.float32)
            X_strat = torch.tensor(np.array(t_strat), dtype=torch.float32)
            X_logicas = torch.tensor(np.array(t_logicas), dtype=torch.float32)
            X_avanzados = torch.tensor(np.array(t_avanzados), dtype=torch.float32)
            X_matriz = torch.tensor(np.array(t_matriz), dtype=torch.float32)
            X_entropia = torch.tensor(np.array(t_entropia), dtype=torch.float32)
            Y = torch.tensor(np.array(t_targets), dtype=torch.float32)

            device = torch.device("cpu")
            modelo = modelo_bio.EncodersMultimodalesUnificados().to(device)
            criterio = torch.nn.MSELoss()
            optimizador = torch.optim.AdamW(modelo.parameters(), lr=0.002, weight_decay=1e-4)

            self.txt_salida_pipeline.insert(tk.END, f"[✔] Secuencias FASTA cargadas: {ids_validos}\n")
            self.txt_salida_pipeline.insert(tk.END, f"[✔] Encoders Unificados, MLPD y Q-MLPD inicializados.\n")
            self.txt_salida_pipeline.insert(tk.END, "[*] Ejecutando entrenamiento del modelo...\n")
            self.root.update_idletasks()

            # Bucle de entrenamiento integrado
            for epoch in range(300):
                modelo.train()
                optimizador.zero_grad()
                predicciones, _ = modelo(X_dna, X_spatial, X_strat, X_logicas, X_avanzados, X_matriz, X_entropia, factor_g=factor_g)
                loss = criterio(predicciones, Y)
                loss.backward()
                optimizador.step()

            self.txt_salida_pipeline.insert(tk.END, f"[✔] ¡Entrenamiento completado! MSE Loss Final: {loss.item():.7f}\n\n")

            # Inferencia, Renderizado 3D y Generación Ex Nihilo del Fenotipo
            modelo.eval()
            with torch.no_grad():
                y_final, Z_representacion = modelo(X_dna, X_spatial, X_strat, X_logicas, X_avanzados, X_matriz, X_entropia, factor_g=factor_g)
                self.txt_salida_pipeline.insert(tk.END, f"[✔] Dimensión del Espacio Latente Z Cuántico: {Z_representacion.shape}\n")
                self.txt_salida_pipeline.insert(tk.END, "--- Resultados de Inferencia por Secuencia ---\n")
                for i, id_s in enumerate(ids_validos):
                    self.txt_salida_pipeline.insert(tk.END, f"  - ID: {id_s} | Predicción: {y_final[i].item():.6f}\n")

                # Generar reporte gráfico 3D automatizado
                self.txt_salida_pipeline.insert(tk.END, "\n[📈] Generando gráfica de deformación topológica 3D ('reporte_cuantico_genomico.png')...\n")
                modelo_bio.simular_y_graficar_espacio_cuantico(Z_representacion.cpu())
                
                # Sintetizar visualmente el fenotipo geométrico
                self.txt_salida_pipeline.insert(tk.END, "[🎨] Sintetizando fenotipo visual geométrico ex nihilo ('fenotipo_genomico_sintetizado.png')...\n")
                z_sample = Z_representacion[0:1]
                matriz_fenotipo = modelo.motor_geometrico.forward_generar_geometria(z_sample, seed=42)
                imagen_fenotipo = Image.fromarray((matriz_fenotipo * 255).astype(np.uint8), 'RGB')
                imagen_fenotipo.save("fenotipo_genomico_sintetizado.png")
                self.txt_salida_pipeline.insert(tk.END, "[✔] ¡Archivos de reporte y fenotipo exportados exitosamente!\n")

            self.lbl_estado.config(text="Estado: Pipeline unificado y renderizado completados con éxito.")
        except Exception as e:
            messagebox.showerror("Error", f"Fallo al ejecutar el pipeline unificado: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AppModeloGenomicoEpica(root)
    root.mainloop()