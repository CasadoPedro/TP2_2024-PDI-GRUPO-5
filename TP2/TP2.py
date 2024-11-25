import cv2
import numpy as np
import matplotlib.pyplot as plt


# Problema 1 - Detección y clasificación de monedas y dados


# Cargar imagen
img = cv2.imread("monedas.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img), plt.title('Imagen original RGB'), plt.axis('off'), plt.show()


# Ejercicio A
# Transformar la imagen a HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
plt.imshow(img_hsv), plt.title('Imagen original HSV'), plt.axis('off'), plt.show()
# Separar los canales H, S y V
h, s, v = cv2.split(img_hsv)
# Mostrar canales separados
plt.figure(figsize=(12, 4))
plt.subplot(2, 2, 1), plt.imshow(h, cmap='hsv'), plt.title("Hue (H)"), plt.axis('off')
plt.subplot(2, 2, 2), plt.imshow(s, cmap='gray'), plt.title("Saturation (S)"), plt.axis('off')
plt.subplot(2, 2, 3), plt.imshow(v, cmap='gray'), plt.title("Value (V)"), plt.axis('off')
plt.show()


# Umbralización del canal H para encontrar monedas
h_monedas = cv2.inRange(h, 10, 30)
plt.imshow(h_monedas, cmap='gray'), plt.title('H umbralado para monedas'), plt.axis('off'), plt.show()
# Encontrar contornos de monedas
contours, _ = cv2.findContours(h_monedas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
monedas = list()
# Iterar sobre los contornos detectados
for contour in contours:
    # Obtener bounding box de cada contorno
    x, y, w, h = cv2.boundingRect(contour)
    # Agrandamos 5px las bounding box para asegurarnos de no perder parte de las monedas
    x -= 5; y -= 5; w += 10; h += 10
    # Umbralizamos por tamaño para limpiar objetos no deseados
    if w > 150 and h > 150:
        # Recortar moneda
        moneda = img[y:y+h, x:x+w]
        monedas.append(moneda)
        # Ver moneda
        plt.imshow(moneda), plt.title(f"Moneda {len(monedas)}"), plt.axis('off'), plt.show()


# Umbralización del canal V para encontrar dados
v_dados = cv2.inRange(v, 180, 220)
plt.imshow(v_dados, cmap='gray'), plt.title('V umbralado para dados'), plt.axis('off'), plt.show()
# Encontrar contornos de dados
contours, _ = cv2.findContours(v_dados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
dados = list()
# Iterar sobre los contornos detectados
for contour in contours:
    # Obtener bounding box de cada contorno
    x, y, w, h = cv2.boundingRect(contour)
    # Agrandamos 5px las bounding box para asegurarnos de no perder parte de los dado
    x -= 5; y -= 5; w += 10; h += 10
    # Umbralizamos por tamaño para limpiar objetos no deseados
    if w > 300 and h > 300:
        # Recortar dado
        dado = img[y:y+h, x:x+w]
        dados.append(dado)
        # Ver dado
        plt.imshow(dado), plt.title(f"Dado {len(dados)}"), plt.axis('off'), plt.show()
       
# Ejercicio B
monedas_clasificadas = list()
for moneda in monedas:
    # Obtener área de la moneda
    h, w = moneda.shape[:2]
    area = h * w
    # Clasificar según área
    valor = 1
    if area < 110000:
        valor = 0.1
    elif area > 130000:
        valor = 0.5
    monedas_clasificadas.append((moneda, valor))
total = 0
total1 = 0
total05 = 0
total01 = 0
for tupla in monedas_clasificadas:
    total += tupla[1]
    if tupla[1] == 1: total1 += 1
    elif tupla[1] == 0.5: total05 += 1
    else: total01 += 1
    # Ver resultados
    plt.imshow(tupla[0]), plt.title(f"Moneda de ${tupla[1]}"), plt.axis('off'), plt.show()
print(f"Cantidad de monedas de $1: {total1}")
print(f"Cantidad de monedas de $0.5: {total05}")
print(f"Cantidad de monedas de $0.1: {total01}")    
print(f"La suma total de dinero en monedas es de: ${total:.2f}")


# Ejercicio C
dados_clasificados = list()
for dado in dados:
    dado_gray = cv2.cvtColor(dado, cv2.COLOR_RGB2GRAY)
    # Umbralizamos y aplicamos filtro para limpiar ruido
    _, dado_binary = cv2.threshold(dado_gray, 50, 255, cv2.THRESH_BINARY_INV)
    dado_clean = cv2.medianBlur(dado_binary, 19)
    # Ver resultado
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(dado_binary, cmap='gray'), axes[0].set_title("Dado binarizado"), axes[0].axis('off')
    axes[1].imshow(dado_clean, cmap='gray'), axes[1].set_title("Dado binarizado + filtrado"), axes[1].axis('off')
    plt.show()
    # Detectar contornos
    circulos = list()
    contours, _ = cv2.findContours(dado_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Validamos que sea un círculo de la cara visible del dado según su área
    for contour in contours:
        area = cv2.contourArea(contour)
        if 1000 < area < 3000:
            circulos.append(contour)
    # Dibujar los contornos válidos en el dado
    dado_circulos = dado.copy()
    cv2.drawContours(dado_circulos, circulos, -1, (0, 255, 0), 2)
    dados_clasificados.append((dado_circulos, len(circulos)))
total = 0    
for tupla in dados_clasificados:    
    total += tupla[1]
    # Ver resultados
    plt.imshow(tupla[0]), plt.title(f"Cara visible: {tupla[1]}"), plt.axis('off'), plt.show()
print(f"Hay {len(dados_clasificados)} dados.")
print(f"La suma de los dados es de: {total}")

# Problema 2 - Detección de patentes


autos = ["img01.png", "img02.png", "img03.png", "img04.png",
         "img05.png", "img06.png", "img07.png", "img08.png",
         "img09.png", "img10.png", "img11.png", "img12.png"]


# Transformar a escala de grises
# Umbralado
# Componentes conectadas
# Filtrado por área
# Componentes conectadas
# Filtro relación aspecto
# Componentes de a tres


for auto in autos:
    # Cargar imagen
    img = cv2.imread(auto)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(img), plt.title(f'{auto} RGB'), plt.axis('off'), fullscreen(),plt.show()


    # Transformar a escala de grises
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(img_gray, cmap='gray'), plt.title(f'{auto} escala de grises'), plt.axis('off'), plt.show()
   
    # Umbralado - Binarización
    _, img_binary = cv2.threshold(img_gray, 124, 255, cv2.THRESH_BINARY)    
    #plt.imshow(img_binary, cmap='gray'), plt.title(f'{auto} binaria'), plt.axis('off'), plt.show()
       
    # Componentes conectadas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_binary, 8, cv2.CV_32S)
    #plt.imshow(labels, cmap='gray'), plt.title(f'{auto} componentes conectadas'), plt.axis('off'), plt.show()
    # Umbralamos las labels según área
    filtered_labels = np.zeros_like(labels)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if 12 <= area <= 98:
            filtered_labels[labels == label] = label
    #plt.imshow(filtered_labels, cmap='gray'), plt.title(f'{auto} filtro por area'), plt.axis('off'), plt.show()
   
    filtered_labels = np.uint8(filtered_labels)
    # Filtro relación de aspecto, ver que los componentes sean más altos que anchos
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filtered_labels, 4, cv2.CV_32S)
    filtered_labels2 = np.zeros_like(labels)
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        aspecto = h / w if w > 0 else 0  # para evitar división por cero
        if 1 < aspecto <= 2.5:  # ajustar
            filtered_labels2[labels == label] = label
    #plt.imshow(filtered_labels2, cmap='gray'), plt.title(f'{auto} filtro relacion de aspecto'), plt.axis('off'), plt.show()
   
    filtered_labels2 = np.uint8(filtered_labels2)
    # Componentes de a tres
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filtered_labels2, 8, cv2.CV_32S)
    trios = np.zeros_like(labels)  
    # Recorremos de a tríos
    for label1 in range(1, num_labels):
        for label2 in range(label1 + 1, num_labels):
            for label3 in range(label2 + 1, num_labels):
                c1, c2, c3 = centroids[label1], centroids[label2], centroids[label3]
                # Verificar alineación vertical
                if abs(c1[1] - c2[1]) <= 12 and abs(c2[1] - c3[1]) <= 12:
                    # Verificar distancias horizontales
                    dist1 = abs(c2[0] - c1[0])
                    dist2 = abs(c3[0] - c2[0])
                    if dist1 <= 30 and dist2 <= 30:
                        # Etiquetar tripletas válidas
                        trios[labels == label1] = 255
                        trios[labels == label2] = 255
                        trios[labels == label3] = 255
    #plt.imshow(trios, cmap='gray'), plt.title(f'{auto} trios'), plt.axis('off'), plt.show()
   
    trios = np.uint8(trios)
    # Recorte de la patente
    x, y, w, h = cv2.boundingRect(trios)
    patente = img[y:y+h, x:x+w]
    plt.imshow(patente, cmap='gray'), plt.title(f'{auto} patente'), plt.axis('off'), plt.show()
   
    # Recorte caracteres individuales
    contours, _ = cv2.findContours(trios, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)
    fig, axes = plt.subplots(1, n, figsize=(10, 6))
    fig.suptitle(f" {auto} caracteres detectados")
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # Recortar la región del carácter
        caracter = img[y-5:y+h+5, x:x+w]
        ax = axes[i]
        ax.imshow(caracter), ax.set_title(f"Caracter {i+1}"), ax.axis('off')
    plt.show()