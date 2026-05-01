import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

if not os.path.exists('./ruidos'):
        os.makedirs('./ruidos')

#validar se as cores estão no range
def valid_color(image):
    x, y, c = image.shape
    for i in range(x):
        for j in range(y):
            for k in range(c):
                if(image[i,j,k] < 0):
                    image[i,j,k] = 0
                elif(image[i,j,k] > 255):
                    image[i,j,k] = 255

#passa o filtro pela imagem inteira
def aplicar_filtro(image, kernel):
    if len(image.shape) == 3:
        image = image[:, :, 0]
    x, y = image.shape
    
    img_pad = np.zeros((x + 2, y + 2)) #imagem com padding (zeros em volta)
    img_pad[1:x+1, 1:y+1] = image
    
    img_saida = np.zeros((x, y))
    
    for i in range(x):
        for j in range(y):
            #passa pela parte da imagem inteira
            vizinhanca = img_pad[i : i+3, j : j+3] 
            
            # Operação de convolução
            valor_calculado = np.sum(vizinhanca * kernel)
            img_saida[i, j] = valor_calculado
            
    return img_saida

#gera o filtro da mediana
def filtro_mediana(image):
    if len(image.shape) == 3: image = image[:,:,0]
    x, y = image.shape
    img_pad = np.zeros((x + 2, y + 2))
    img_pad[1:x+1, 1:y+1] = image
    img_saida = np.zeros((x, y))
    
    for i in range(x):
        for j in range(y):
            vizinhanca = img_pad[i : i+3, j : j+3].flatten()
            img_saida[i, j] = np.median(vizinhanca)
    return img_saida

#ruido preto e branco
def ruido_sal_pimenta(image, porcentagem):
    img_ruido = image.copy()
    if len(img_ruido.shape) == 2:
        img_ruido = cv2.merge([img_ruido, img_ruido, img_ruido])
    x, y, c = img_ruido.shape
    total_pixels = int((porcentagem / 100) * (x * y))
    for _ in range(total_pixels):
        i, j = random.randint(0, x-1), random.randint(0, y-1)
        img_ruido[i, j, :] = 255 if random.random() > 0.5 else 0
    return img_ruido

#ruido aleatorio
def ruido_uniforme(image, porcentagem):
    img_ruido = image.copy()
    if len(img_ruido.shape) == 2:
        img_ruido = cv2.merge([img_ruido, img_ruido, img_ruido])
    x, y, c = img_ruido.shape
    total_pixels = int((porcentagem / 100) * (x * y))
    for _ in range(total_pixels):
        i, j = random.randint(0, x-1), random.randint(0, y-1)
        img_ruido[i, j, :] = random.randint(0, 255)
    return img_ruido


path = "./ruidos/original.jpg"

img_orig = cv2.imread(path)
'''
#deixa grayscale
R = img_orig[:, :, 2]
G = img_orig[:, :, 1]
B = img_orig[:, :, 0]

gray = (gR + gG + gB).astype(np.uint8)
'''
img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)


#ruidos com 15% de intensidade
img_sp = ruido_sal_pimenta(img_gray, 15) 
img_un = ruido_uniforme(img_gray, 15)
cv2.imwrite("./ruidos/ruido-sp(15%).png", img_sp)
cv2.imwrite("./ruidos/ruido-un(15%).png", img_un)

kernel_media = np.ones((3,3)) / 9   
img_media_sp = aplicar_filtro(img_sp, kernel_media) 
img_media_un = aplicar_filtro(img_un, kernel_media) 
cv2.imwrite("./ruidos/media-sp.png", img_media_sp)
cv2.imwrite("./ruidos/media-un.png", img_media_un)

img_mediana_sp = filtro_mediana(img_sp)
img_mediana_un = filtro_mediana(img_un)
cv2.imwrite("./ruidos/mediana-sp.png", img_mediana_sp)
cv2.imwrite("./ruidos/mediana-un.png", img_mediana_un)

k_ponto = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
img_ponto = aplicar_filtro(img_gray, k_ponto)
cv2.imwrite("./ruidos/pontos.png", img_ponto)

k_lin_h = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]])
img_lin_h = aplicar_filtro(img_gray, k_lin_h)
cv2.imwrite("./ruidos/linhas-horizontais.png", img_lin_h)

k_lin_v = np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]])
img_lin_v = aplicar_filtro(img_gray, k_lin_v)
cv2.imwrite("./ruidos/linhas-verticais.png", img_lin_v)

img_bordas = np.sqrt(np.square(img_lin_h) + np.square(img_lin_v))
valid_color(img_bordas.reshape(img_bordas.shape + (1,)))
cv2.imwrite("./ruidos/bordas.png", img_bordas)

