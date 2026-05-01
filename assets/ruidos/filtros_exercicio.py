import cv2
import numpy as np
import matplotlib.pyplot as plt

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

#binariza a imagem de acordo com um limiar
def limiar_image(image, limiar):
    x, y, c = image.shape
    image_bin = np.zeros((x, y, c))
    for i in range(x):
        for j in range(y):
            for k in range(c):
                if (image[i, j, k] < limiar):
                    image_bin[i, j, k] = 0
                elif (image[i, j, k] >= limiar):
                    image_bin[i, j, k] = 255
    return image_bin

#encontra o menor e maior pixel da imagem (intensidade)
def min_max_value(image):
    x, y, c = image.shape
    max_val = 0
    min_val = 255
    for i in range(x):
        for j in range(y):
            for k in range(c):
                if(image[i,j,k] < min_val):
                    min_val = image[i,j,k]
                if(image[i,j,k] > max_val):
                    max_val = image[i,j,k]
    return min_val, max_val

#Alargamento de histograma
def alarg_image(image):
    min_val, max_val = min_max_value(image)
    x, y, c = image.shape
    image_alarg = np.zeros((x, y, c))
    for i in range(x):
        for j in range(y):
            for k in range(c):
                image_alarg[i,j,k] = (image[i,j,k] - min_val) * (255 / (max_val - min_val))
    valid_color(image_alarg)
    return image_alarg

#Quantização
def quant_image(image, tons, inter):
    x, y, c = image.shape
    quant_img = np.zeros((x, y, c))
    quant = np.zeros((x, y, c))
    for i in range(x):
        for j in range(y):
            for k in range(c):
                quant[i,j,k] = image[i,j,k] // inter
                quant_img[i, j, k] = int(((quant[i,j,k] * inter)) - (inter / 2))
    return quant_img, quant

#Histograma
def gerar_hist(image, tons, path):
    bins = np.arange(tons)
    contagens = np.zeros(tons)

    for valor in image[:, :, 0].ravel():
        contagens[int(valor)] += 1

    if(tons > 10):
        dados = [b for b in bins if contagens[b] > 0]
    else:
        dados = bins

    plt.figure(figsize=(6, 4))
    plt.bar(bins, contagens, color='#4285F4', width=0.7)
    plt.title("Histograma da Imagem Quantizada")
    plt.xlabel("Nível de Intensidade (Tom)")
    plt.ylabel("Quantidade de Pixels")
    plt.xticks(dados)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(path)
    plt.close()

    return contagens

#Equalização de histograma
def equaliz_image(dados, tons):
    prob = np.zeros(tons)
    prob_sum = np.zeros(tons)
    prob_mult = np.zeros(tons)
    nova_intens = np.zeros(tons)
    tot = 0

    for d in range(tons):
        tot += dados[d]

    for p in range(tons):
        prob[p] = dados[p] / tot

    for s in range(tons):
        if(s == 0):
            prob_sum[s] = prob[s]
        else:
            prob_sum[s] = prob_sum[s-1] + prob[s]

    for m in range(tons):
        prob_mult[m] = prob_sum[m] * (tons - 1)
        nova_intens[m] = np.floor(prob_mult[m] + 0.5)

    print(f"Q = {dados}")
    print(f"P = {prob}")
    print(f"Ac = {prob_sum}")
    print(f"M = {prob_mult}")
    print(f"Ar = {nova_intens}")

    return nova_intens

def aplicar_equaliz(image_quant, tabela_mapeamento):
    x, y, c = image_quant.shape
    img_equaliz = np.zeros((x, y, c))

    for i in range(x):
        for j in range(y):
            for k in range(c):
                valor_antigo = int(image_quant[i, j, k])
                img_equaliz[i, j, k] = tabela_mapeamento[valor_antigo]

    return img_equaliz

def aplicar_filtro(image, kernel):
    x, y = image.shape

    img_pad = np.zeros((x + 2, y + 2))
    img_pad[1:x+1, 1:y+1] = image

    img_saida = np.zeros((x, y))

    for i in range(x):
        for j in range(y):
            vizinhanca = img_pad[i : i+3, j : j+3]
            valor_calculado = np.sum(vizinhanca * kernel)
            img_saida[i, j] = valor_calculado

    return img_saida

# distância a partir do centro
def gerar_matriz_distancia(lin, col):
    dist = np.zeros((lin, col))
    cy, cx = lin // 2, col // 2

    for i in range(lin):
        for j in range(col):
            distancia = np.sqrt((i - cy)**2 + (j - cx)**2)
            dist[i, j] = distancia
    return dist

def freq_mantida(matriz, d_zero, baixa_alta):
    x, y = matriz.shape
    norm = np.zeros((x, y))

    for i in range(x):
        for j in range(y):
            if(matriz[i,j] <= d_zero):
                if(baixa_alta == 'baixa'):
                    norm[i,j] = 1
                elif(baixa_alta == 'alta'):
                    norm[i,j] = 0
            else:
                if(baixa_alta == 'baixa'):
                    norm[i,j] = 0
                elif(baixa_alta == 'alta'):
                    norm[i,j] = 1
    return norm


R = np.array([
    [255,255,0,255,255],
    [255,255,255,255,255],
    [0,255,255,255,0],
    [0,255,255,255,0]
], dtype=np.uint8)

G = np.array([
    [0,0,255,0,0],
    [255,255,255,255,255],
    [255,255,255,255,255],
    [0,255,255,255,0]
], dtype=np.uint8)

B = np.array([
    [0,255,0,255,0],
    [0,255,255,255,0],
    [255,255,255,255,255],
    [255,255,255,255,255]
], dtype=np.uint8)

img = cv2.merge([B,G,R])
img_large = cv2.resize(img, (500,400), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("./imagem.png", img_large)

# Grayscale
gR = R * 0.299
gG = G * 0.587
gB = B * 0.114

gray = (gR + gG + gB).astype(np.uint8)

img_gray = cv2.merge([gray, gray, gray])
valid_color(img_gray)

gray_large = cv2.resize(img_gray, (500,400), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("./gray.png", gray_large)

# Binarização
img_bin = limiar_image(img_gray, 230)
bin_large = cv2.resize(img_bin, (500,400), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("./bin.png", bin_large)

# Alargamento
img_alarg = alarg_image(img_gray)
alarg_large = cv2.resize(img_alarg, (500,400), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("./alarg.png", alarg_large)

# Quantização
tons = 8
inter = 256 / tons
img_quant, quant = quant_image(img_gray, tons, inter)
quant_large = cv2.resize(img_quant, (500,400), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("./quant.png", quant_large)

# Histograma
cont = gerar_hist(quant, tons, "./hist(0-7).png")
cont_img = gerar_hist(img_quant, 255, "./hist(0-255).png")

# Equalização
nova_intens = equaliz_image(cont, tons)
img_equaliz = aplicar_equaliz(quant, nova_intens)
gerar_hist(img_equaliz, tons, "./hist_equaliz.png")
equaliz_large = cv2.resize(img_equaliz, (500,400), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("./equaliz.png", equaliz_large)

# Filtro de média (passa-baixa) - Domínio espacial
kernel_media = np.ones((3, 3)) / 9
img_passabaixa = aplicar_filtro(gray, kernel_media)
valid_color(img_passabaixa.reshape(img_passabaixa.shape + (1,)))

baixa_large = cv2.resize(img_passabaixa, (500, 400), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("./passa-baixa.png", baixa_large)

# Filtro Laplaciano (passa-alta) - Domínio espacial
kernel_laplaciano = np.array([
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0]
])
img_passalta = aplicar_filtro(gray, kernel_laplaciano)
valid_color(img_passalta.reshape(img_passalta.shape + (1,)))

alta_large = cv2.resize(img_passalta, (500, 400), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("./passa-alta.png", alta_large)

# Fourier
#fft2 para transforma em 2D
dom_freq = np.fft.fft2(gray) #é equivalente a função do F(u,v)
dom_freq_shift = np.fft.fftshift(dom_freq)  # Desloca DC para o centro

i, j = gray.shape
dist = gerar_matriz_distancia(i, j)

# Passa-baixa (domínio da frequência)
norm = freq_mantida(dist, 1, 'baixa')

filtro_baixo = dom_freq_shift * norm
filtro_baixo_ishift = np.fft.ifftshift(filtro_baixo)
baixa_freq = np.fft.ifft2(filtro_baixo_ishift) #filtro vira a imagem de volta
baixa_freq = np.abs(baixa_freq) #eliminar números complexos
valid_color(baixa_freq.reshape(baixa_freq.shape + (1,)))

baixa_freq_large = cv2.resize(baixa_freq, (500, 400), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("./passa-baixa(freq).png", baixa_freq_large)

# Passa-alta (domínio da frequência)
norm_alta = freq_mantida(dist, 1, 'alta')

filtro_alto = dom_freq_shift * norm_alta
filtro_alto_ishift = np.fft.ifftshift(filtro_alto)
alta_freq = np.fft.ifft2(filtro_alto_ishift)
alta_freq = np.abs(alta_freq)
valid_color(alta_freq.reshape(alta_freq.shape + (1,)))

alta_freq_large = cv2.resize(alta_freq, (500, 400), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("./passa-alta(freq).png", alta_freq_large)

cv2.waitKey(0)
cv2.destroyAllWindows()