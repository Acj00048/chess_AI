import torch
import chess
# Importamos la V3 de tu entrenador
from entrenador import MagnusNet, AjedrezVocab 

def cargar_cerebro(ruta_modelo):
    print("🧠 Despertando al Asistente Magnus V4...")
    checkpoint = torch.load(ruta_modelo, map_location=torch.device('cpu'))
    
    vocab = AjedrezVocab()
    vocab.move_to_id = checkpoint['vocab']
    vocab.id_to_move = {v: k for k, v in vocab.move_to_id.items()}
    vocab_size = len(vocab.move_to_id)
    
    # IMPORTANTE: Usamos las dimensiones de la V3
    modelo = MagnusNet(vocab_size, emb_dim=256, hidden_dim=512)
    modelo.load_state_dict(checkpoint['model_state'])
    modelo.eval() 
    return modelo, vocab

def predecir_jugada_ia(modelo, vocab, historial_jugadas, tablero_actual):
    # Cogemos las últimas 9 jugadas (para dejar hueco al token de resultado)
    tokens = [vocab.move_to_id.get(j, 1) for j in historial_jugadas[-9:]]
    
    # LA MAGIA DE LA V3: Inyectamos el token <WIN> (ID = 2) al principio siempre
    secuencia_con_intencion = [2] + tokens
    
    entrada = torch.tensor([secuencia_con_intencion])
    
    with torch.no_grad():
        salida = modelo(entrada)
        
    probabilidades = torch.argsort(salida[0], descending=True)
    
    for token_id in probabilidades:
        jugada_texto = vocab.id_to_move.get(token_id.item(), "<UNK>")
        if jugada_texto not in ["<PAD>", "<UNK>", "<WIN>", "<LOSS>", "<DRAW>"]:
            try:
                movimiento = tablero_actual.parse_san(jugada_texto)
                if movimiento in tablero_actual.legal_moves:
                    return jugada_texto
            except ValueError:
                continue
    return None

def iniciar_asistente():
    modelo, vocab = cargar_cerebro('magnus_brain_v4.pth')
    tablero = chess.Board()
    historial = []
    
    print("\n" + "="*50)
    print(" 🎧 MAGNUS-NET V4: MODO ASISTENTE CENTAURO 🎧")
    print("="*50)
    
    color_input = input("¿Con qué color vas a jugar en la web? (b/n): ").lower()
    mi_color = chess.WHITE if color_input == 'b' else chess.BLACK
    
    while not tablero.is_game_over():
        print("\n" + str(tablero) + "\n")
        
        if tablero.turn == mi_color:
            print("🤖 Magnus-Net analizando vías de victoria...")
            jugada_sugerida = predecir_jugada_ia(modelo, vocab, historial, tablero)
            
            if jugada_sugerida:
                print(f"💡 SUGERENCIA ESTRATÉGICA: Juega >> {jugada_sugerida} <<")
                decision = input("Pulsa Enter si has jugado eso, o escribe tu jugada alternativa: ")
                
                jugada_final = jugada_sugerida if decision == "" else decision
                try:
                    tablero.push_san(jugada_final)
                    historial.append(jugada_final)
                except ValueError:
                    print("❌ Escribiste una jugada inválida. El tablero se ha desincronizado.")
                    break
            else:
                print("❌ La IA está confundida (Fuera de Distribución).")
                break
                
        else:
            jugada_rival = input("👉 ¿Qué acaba de jugar tu rival en la web?: ")
            if jugada_rival.lower() == 'salir': break
            
            try:
                tablero.push_san(jugada_rival)
                historial.append(jugada_rival)
            except ValueError:
                print("❌ Jugada inválida. Revisa bien la notación (ej: Nf3, O-O, exd5).")

    print("\n🏆 FIN DE LA PARTIDA 🏆")

if __name__ == "__main__":
    iniciar_asistente()