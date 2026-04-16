import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- 1. CONFIGURACIÓN DEL MOTOR M4 ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- 2. VOCABULARIO CON TOKENS DE CONTROL ---
class AjedrezVocab:
    def __init__(self):
        # Añadimos los tokens de resultado
        self.move_to_id = {"<PAD>": 0, "<UNK>": 1, "<WIN>": 2, "<LOSS>": 3, "<DRAW>": 4}
        self.id_to_move = {v: k for k, v in self.move_to_id.items()}
        self.n_moves = 5

    def construir_vocab(self, lista_partidas):
        for partida in lista_partidas:
            if isinstance(partida, str):
                for jugada in partida.split():
                    if jugada not in self.move_to_id:
                        self.move_to_id[jugada] = self.n_moves
                        self.id_to_move[self.n_moves] = jugada
                        self.n_moves += 1
        return self.n_moves

# --- 3. DATASET CON CONTEXTO DE RESULTADO ---
class MagnusDataset(Dataset):
    def __init__(self, df, vocab, seq_len=60):
        self.datos = []
        self.objetivos = []
        
        for _, row in df.iterrows():
            partida = row['moves']
            resultado = str(row['result']) # Necesitamos la columna 'result'
            
            # Asignamos el token de contexto según el resultado
            if "1-0" in resultado: res_token = 2 # Ganan blancas
            elif "0-1" in resultado: res_token = 3 # Ganan negras
            else: res_token = 4 # Tablas
            
            if isinstance(partida, str):
                tokens = [vocab.move_to_id.get(j, 1) for j in partida.split()]
                for i in range(len(tokens) - seq_len):
                    # La secuencia ahora EMPIEZA con el token de resultado
                    secuencia = [res_token] + tokens[i:i+seq_len]
                    self.datos.append(torch.tensor(secuencia))
                    self.objetivos.append(tokens[i+seq_len])

    def __len__(self): return len(self.datos)
    def __getitem__(self, idx): return self.datos[idx], self.objetivos[idx]

# --- 4. ARQUITECTURA "PRO" (V3) ---
class MagnusNet(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden_dim=512):
        super(MagnusNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        
        # LayerNorm ayuda a que el modelo aprenda más rápido y estable
        self.ln = nn.LayerNorm(emb_dim)
        
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, vocab_size)
        
        self.relu = nn.GELU() # Una activación más moderna que ReLU
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x)
        x = self.ln(x)
        x = torch.mean(x, dim=1) 
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x)) # Nueva capa
        x = self.fc4(x)
        return x

# --- 5. ENTRENAMIENTO VITAMINADO ---
def entrenar():
    print("📦 Cargando datos de Magnus...")
    df = pd.read_csv('magnus_games.csv')
    
    vocab = AjedrezVocab()
    vocab_size = vocab.construir_vocab(df['moves'].tolist()[:17000]) # Subimos a 17.000 partidas

    dataset = MagnusDataset(df.head(17000), vocab)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    modelo = MagnusNet(vocab_size).to(device)
    # Bajamos el Learning Rate para que no "salte" y encuentre el error mínimo
    optimizer = optim.AdamW(modelo.parameters(), lr=0.0003, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    print(f"🧠 Entrenando V3 (Contexto de Resultado) en {device}...")
    
    for epoch in range(20): # Más épocas para dejar que el LayerNorm trabaje
        total_loss = 0
        modelo.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            salida = modelo(x)
            loss = criterion(salida, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Época {epoch+1}/20 - Error: {total_loss/len(loader):.4f}")

    torch.save({'model_state': modelo.state_dict(), 'vocab': vocab.move_to_id}, 'magnus_brain_v4.pth')
    print("💾 ¡V4 Guardada con éxito!")

if __name__ == "__main__":
    entrenar()