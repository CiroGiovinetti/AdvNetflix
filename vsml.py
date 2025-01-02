import warnings
import cv2
import torch
from segment_anything import SamPredictor, sam_model_registry
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import csv
import webbrowser
import time

# Disabilita i warning FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore")

# Percorso al modello pre-addestrato (modifica se necessario)
model_path = 'sam_vit_b_01ec64.pth'  # Inserisci il nome del tuo file di checkpoint

# Inizializza SAM con il modello pre-addestrato
sam = sam_model_registry["vit_b"](checkpoint=model_path)
sam.to("cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)

# Carica il modello pre-addestrato (ResNet50) senza l'ultimo strato di classificazione
modello = models.resnet50(pretrained=True)
modello = torch.nn.Sequential(*list(modello.children())[:-1])  # Rimuovi l'ultimo layer
modello.eval()  # Metti il modello in modalità di valutazione

# Trasformazioni per adattare le immagini al modello
trasformazioni = transforms.Compose([
    transforms.Resize((224, 224)),  # Dimensione standard per ResNet50
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Funzione per estrarre il vettore delle caratteristiche da un'immagine
def estrai_caratteristiche(immagine_path):
    immagine = Image.open(immagine_path).convert('RGB')
    immagine = trasformazioni(immagine).unsqueeze(0)  # Aggiungi dimensione batch
    with torch.no_grad():
        caratteristiche = modello(immagine)
    return caratteristiche.flatten()

# Funzione per trovare la corrispondenza del campione nella galleria
def trova_cappotto(campione_path, galleria_dir, soglia=0.75):
    campione_feat = estrai_caratteristiche(campione_path)

    trovato = False  # Variabile booleana per indicare se l'immagine campione è stata trovata nella galleria
    nome_file_trovato = None  # Variabile per memorizzare il nome del file trovato

    # Scansiona la galleria
    for nome_file in os.listdir(galleria_dir):
        if nome_file.endswith(('.jpg', '.png')):
            percorso_immagine_galleria = os.path.join(galleria_dir, nome_file)
            galleria_feat = estrai_caratteristiche(percorso_immagine_galleria)

            # Calcola la similarità usando il prodotto scalare
            similarità = torch.cosine_similarity(campione_feat, galleria_feat, dim=0).item()

            # Stampa la similarità per ogni immagine nella galleria
            print(f"Similarità con {nome_file}: {similarità:.2f}")  # Stampa la similarità

            # Verifica se la similarità supera la soglia
            if similarità >= soglia:
                trovato = True
                nome_file_trovato = nome_file
                break  # Esci dal ciclo se trovato

    # Stampa il risultato finale
    if trovato:
        print(f"{nome_file_trovato} è l'immagine che cercavi, la sua similarità supera 0.75")
        link = trova_link_per_file(nome_file_trovato)
        if link:
            webbrowser.open(link)  # Apri il link nel browser
            print(f"Link del prodotto {nome_file_trovato}:\n{link}")
        else:
            print(f"Nessun link trovato per {nome_file_trovato}")
    else:
        print("Non trovato")

# Funzione per trovare il link corrispondente al file immagine nel CSV
def trova_link_per_file(nome_file):
    # Percorso del file CSV
    csv_path = 'link_galleria.csv'  # Assicurati che il percorso sia corretto
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Supponiamo che la struttura del CSV sia: nome_file, url
            if row[0] == nome_file:
                return row[1]  # Restituisce l'URL
    return None  # Se non trovata, restituisce None

# Funzione per segmentare l'oggetto con SAM
def segment_object_with_sam(image, click_point, output_path="oggetto_segmentato.png"):
    predictor.set_image(image)

    # Converti le coordinate del clic in un tipo appropriato per il modello
    input_point = torch.tensor([click_point], dtype=torch.float32)  # Aggiungi il tipo float32
    input_label = torch.tensor([1], dtype=torch.int64)  # 1 significa che è un punto positivo

    # Prepara il modello per la previsione
    masks, _, _ = predictor.predict(point_coords=input_point.numpy(), point_labels=input_label.numpy(), multimask_output=False)

    # Crea un'immagine segmentata per l'oggetto selezionato
    mask = masks[0].astype("uint8") * 255  # Rimuovi il .cpu() e usa direttamente numpy
    segmented_object = cv2.bitwise_and(image, image, mask=mask)

    # Crea un'immagine RGBA dove il canale alfa è 255 (opaco) per l'oggetto segmentato
    rgba_segmented = cv2.cvtColor(segmented_object, cv2.COLOR_BGR2BGRA)  # Aggiungi canale alfa
    rgba_segmented[:, :, 3] = mask  # Imposta la trasparenza in base alla maschera: 255 dove l'oggetto, 0 dove non c'è

    # Salva l'immagine segmentata con trasparenza come PNG
    result = Image.fromarray(rgba_segmented)
    result.save(output_path, format="PNG")  # Usa il formato PNG
    print(f"Immagine segmentata salvata in: {output_path}")

# Funzione di callback per il clic del mouse
def on_mouse_click(event, x, y, flags, param):
    global paused, start_time, frame, stop_video

    if event == cv2.EVENT_LBUTTONDOWN:
        # Solo quando il video è in pausa e se il clic è prolungato per 3 secondi
        start_time = time.time()
    
    elif event == cv2.EVENT_LBUTTONUP:
        # Se il clic dura 3 secondi o più, esegui l'operazione
        if time.time() - start_time >= 1 and paused:
            print("Clic prolungato rilevato! Salvataggio immagine.")
            salva_immagine_normale(frame)
            
            # Segmenta l'oggetto sull'immagine normale
            segment_object_with_sam(frame, (x, y))

            # Esegui il confronto dell'oggetto segmentato con la galleria
            trova_cappotto("oggetto_segmentato.png", "galleria")

            # Ferma il video dopo il click prolungato
            paused = True

# Funzione per salvare l'immagine normale dal frame corrente come .jpg
def salva_immagine_normale(frame, output_path="immagine.jpg"):
    cv2.imwrite(output_path, frame)
    print(f"Immagine salvata in: {output_path}")

# Apri il video
cap = cv2.VideoCapture("video.mp4")

if not cap.isOpened():
    print("Errore nell'apertura del video.")
    exit()

# Ottieni il frame rate per la riproduzione alla velocità corretta
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)
paused = False
stop_video = False  # Variabile per fermare il video

# Riproduci il video
while cap.isOpened():
    if not paused and not stop_video:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video", frame)

        # Imposta il callback del mouse *dopo* aver creato la finestra
        cv2.setMouseCallback("Video", on_mouse_click, param=(frame,))

    # Se il video è in pausa, mostriamo l'ultimo frame
    if paused:
        cv2.imshow("Video", frame)

    # Controlla i tasti per la pausa o la chiusura del video
    key = cv2.waitKey(delay) & 0xFF
    if key == ord("q"):
        break  # Uscita dal loop del video con 'q'
    elif key == ord("p"):  # Metti in pausa il video
        paused = True
    elif key == ord("o"):  # Riprendi il video con 'o'
        paused = False


# Rilascia la risorsa video e chiudi tutte le finestre
cap.release()
cv2.destroyAllWindows()
