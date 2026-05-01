import cv2
import numpy as np
import matplotlib as plt

# Definindo as matrizes 4x5 para cada canal 
# Canal vermelho
R = np.array([
    [255, 255,   0, 255, 255],
    [255, 255, 255, 255, 255],
    [  0, 255, 255, 255,   0],
    [  0, 255, 255, 255,   0]
], dtype=np.uint8)

plt.imshow(R)
plt.title("Teste Matriz Red")
plt.colorbar()
plt.show()

# Canal Verde
G = np.array([
    [  0,   0, 255,   0,   0],
    [255, 255, 255, 255, 255],
    [255, 255, 255, 255, 255],
    [  0, 255, 255, 255,   0]
], dtype=np.uint8)

# Canal Azul
B = np.array([
    [  0, 255,   0, 255,   0],
    [  0, 255, 255, 255,   0],
    [255, 255, 255, 255, 255],
    [255, 255, 255, 255, 255]
], dtype=np.uint8)

img = cv2.merge([B, G, R]) #Função merge serve pra unir as matriz em uma unica imagem (o cv2 usa BGR e não RGB)
x, y = R.shape
Z = np.zeros((x,y))
R = cv2.merge([Z,Z,R])

img_large = cv2.resize(img, (500, 400), interpolation=cv2.INTER_NEAREST) #escalar a imagem pq 4x5 é mt pequeno
#"interpolation=cv2.INTER_NEAREST" serve para q a imagem escalada mantenha a mesma ideia dos valores dos pixels só q maior

R_large = cv2.resize(R, (500, 400), interpolation=cv2.INTER_NEAREST)
G_large = cv2.resize(G, (500, 400), interpolation=cv2.INTER_NEAREST) 
B_large = cv2.resize(B, (500, 400), interpolation=cv2.INTER_NEAREST) 

cv2.imwrite("imagem_R.png", R_large)
cv2.imwrite("imagem_G.png", G_large)
cv2.imwrite("imagem_B.png", B_large)

cv2.imwrite("imagem_exercicio.png", img) # Salva a imagem (4x5)


cv2.waitKey(0)
cv2.destroyAllWindows()