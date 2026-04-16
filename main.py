import torch

def comprobar_motor_m4():
    print("🤖 Iniciando diagnóstico del sistema de IA...")
    
    # Comprobamos si el Apple Silicon GPU (MPS) está disponible
    if torch.backends.mps.is_available():
        dispositivo = torch.device("mps")
        print("✅ ¡Aceleración Apple Silicon (MPS) ACTIVADA!")
        print("Tu Mac M4 está listo para entrenar Redes Neuronales a máxima velocidad.")
    else:
        dispositivo = torch.device("cpu")
        print("⚠️ MPS no detectado. Usando CPU normal (será más lento).")
        
    return dispositivo

if __name__ == "__main__":
    device = comprobar_motor_m4()