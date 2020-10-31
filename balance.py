import os
import cv2
from skimage.util import random_noise
from sklearn.model_selection import train_test_split
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
import random

def dividirEBalancearPorClasse(n_classes_balanceamento, test_size):
    pasta_ler = '..\\Base por classe\\'
    classes = os.listdir(pasta_ler) #classes
    X_treinamento_bal, y_treinamento_bal, X_validacao_bal, y_validacao_bal, X_teste_bal, y_teste_bal = [], [], [], [], [], []
    
    x = 0 #ASCH
    pasta_ler2 = pasta_ler + classes[x] + '\\'
    nomes_imgs = os.listdir(pasta_ler2) #classes
    imgs_treinamento, imgs_teste = train_test_split(nomes_imgs, test_size=test_size)
    X_treinamento, y_treinamento, X_teste, y_teste = dataaugmentationASCH(imgs_treinamento, imgs_teste, n_classes_balanceamento, pasta_ler2)
    X_treino, X_validacao, y_treino, y_validacao = train_test_split(X_treinamento, y_treinamento, test_size=test_size)
    X_treinamento_bal = X_treinamento_bal + X_treino
    y_treinamento_bal = y_treinamento_bal + y_treino
    X_validacao_bal = X_validacao_bal + X_validacao
    y_validacao_bal = y_validacao_bal + y_validacao
    X_teste_bal = X_teste_bal + X_teste
    y_teste_bal = y_teste_bal + y_teste
    
    x = 1 #ASCUS
    pasta_ler2 = pasta_ler + classes[x] + '\\'
    nomes_imgs = os.listdir(pasta_ler2) #classes
    imgs_treinamento, imgs_teste = train_test_split(nomes_imgs, test_size=test_size)
    X_treinamento, y_treinamento, X_teste, y_teste = dataaugmentationASCUS(imgs_treinamento, imgs_teste, n_classes_balanceamento, pasta_ler2)
    X_treino, X_validacao, y_treino, y_validacao = train_test_split(X_treinamento, y_treinamento, test_size=test_size)
    X_treinamento_bal = X_treinamento_bal + X_treino
    y_treinamento_bal = y_treinamento_bal + y_treino
    X_validacao_bal = X_validacao_bal + X_validacao
    y_validacao_bal = y_validacao_bal + y_validacao
    X_teste_bal = X_teste_bal + X_teste
    y_teste_bal = y_teste_bal + y_teste
    
    x = 2 #CA
    pasta_ler2 = pasta_ler + classes[x] + '\\'
    nomes_imgs = os.listdir(pasta_ler2) #classes
    imgs_treinamento, imgs_teste = train_test_split(nomes_imgs, test_size=test_size)
    X_treinamento, y_treinamento, X_teste, y_teste = dataaugmentationCA(imgs_treinamento, imgs_teste, n_classes_balanceamento, pasta_ler2)
    X_treino, X_validacao, y_treino, y_validacao = train_test_split(X_treinamento, y_treinamento, test_size=test_size)
    X_treinamento_bal = X_treinamento_bal + X_treino
    y_treinamento_bal = y_treinamento_bal + y_treino
    X_validacao_bal = X_validacao_bal + X_validacao
    y_validacao_bal = y_validacao_bal + y_validacao
    X_teste_bal = X_teste_bal + X_teste
    y_teste_bal = y_teste_bal + y_teste
    
    x = 3 #HSIL
    pasta_ler2 = pasta_ler + classes[x] + '\\'
    nomes_imgs = os.listdir(pasta_ler2) #classes
    imgs_treinamento, imgs_teste = train_test_split(nomes_imgs, test_size=test_size)
    X_treinamento, y_treinamento, X_teste, y_teste = dataaugmentationHSIL(imgs_treinamento, imgs_teste, n_classes_balanceamento, pasta_ler2)
    X_treino, X_validacao, y_treino, y_validacao = train_test_split(X_treinamento, y_treinamento, test_size=test_size)
    X_treinamento_bal = X_treinamento_bal + X_treino
    y_treinamento_bal = y_treinamento_bal + y_treino
    X_validacao_bal = X_validacao_bal + X_validacao
    y_validacao_bal = y_validacao_bal + y_validacao
    X_teste_bal = X_teste_bal + X_teste
    y_teste_bal = y_teste_bal + y_teste
    
    x = 4 #LSIL
    pasta_ler2 = pasta_ler + classes[x] + '\\'
    nomes_imgs = os.listdir(pasta_ler2) #classes
    imgs_treinamento, imgs_teste = train_test_split(nomes_imgs, test_size=test_size)
    X_treinamento, y_treinamento, X_teste, y_teste = dataaugmentationLSIL(imgs_treinamento, imgs_teste, n_classes_balanceamento, pasta_ler2)
    X_treino, X_validacao, y_treino, y_validacao = train_test_split(X_treinamento, y_treinamento, test_size=test_size)
    X_treinamento_bal = X_treinamento_bal + X_treino
    y_treinamento_bal = y_treinamento_bal + y_treino
    X_validacao_bal = X_validacao_bal + X_validacao
    y_validacao_bal = y_validacao_bal + y_validacao
    X_teste_bal = X_teste_bal + X_teste
    y_teste_bal = y_teste_bal + y_teste
    
    x = 5 #Normal
    pasta_ler2 = pasta_ler + classes[x] + '\\'
    nomes_imgs = os.listdir(pasta_ler2) #classes
    imgs_treinamento, imgs_teste = train_test_split(nomes_imgs, test_size=test_size)
    X_treinamento, y_treinamento, X_teste, y_teste = dataaugmentationNormal(imgs_treinamento, imgs_teste, n_classes_balanceamento, pasta_ler2)
    X_treino, X_validacao, y_treino, y_validacao = train_test_split(X_treinamento, y_treinamento, test_size=test_size)
    X_treinamento_bal = X_treinamento_bal + X_treino
    y_treinamento_bal = y_treinamento_bal + y_treino
    X_validacao_bal = X_validacao_bal + X_validacao
    y_validacao_bal = y_validacao_bal + y_validacao
    X_teste_bal = X_teste_bal + X_teste
    y_teste_bal = y_teste_bal + y_teste
    
    return X_treinamento_bal, y_treinamento_bal, X_validacao_bal, y_validacao_bal, X_teste_bal, y_teste_bal

def dataaugmentationLSIL(imagens_treinamento, imagens_teste, n_classes, pasta_classe):
    #6 classes -> 220 imagens viram +1
    #3 classes -> cada imagem vira +1, 100 imagens viram +1
    #2 classes -> 220 imagens viram +1
    n_imagens = len(imagens_treinamento)
    X_treinamento = []
    y_treinamento = []
    X_teste = []
    y_teste = []
    if(n_classes == 6):
        classe = 4 #LSIL
    elif(n_classes == 2):
        classe = 0
    else:
        classe = 1
    for x in range(0,n_imagens):
        img = cv2.imread(pasta_classe + imagens_treinamento[x])
        X_treinamento.append(img)
        y_treinamento.append(classe)
    if (n_classes == 3):
        imagens_operacao = random.sample(range(0,n_imagens), n_imagens)
        for pos in range(0,len(imagens_operacao)):
            if(pos < 100):
                operacoes = random.sample(range(1,10), 2)
            else:
                operacoes = random.sample(range(1,10), 1)
            for operacao in operacoes:
                nova_img = operacaoAugmentation_retornar(operacao, X_treinamento[pos])
                X_treinamento.append(nova_img)
                y_treinamento.append(classe)
    if (n_classes == 6 or n_classes == 2):
        imagens_operacao = random.sample(range(0,n_imagens), 220)
        for imgs_op in imagens_operacao:
            operacao = random.randint(1,10)
            nova_img = operacaoAugmentation_retornar(operacao, X_treinamento[imgs_op])
            X_treinamento.append(nova_img)
            y_treinamento.append(classe)
    for x in range(0,len(imagens_teste)):
        X_teste.append(cv2.imread(pasta_classe + imagens_teste[x]))
        y_teste.append(classe)
    return X_treinamento, y_treinamento, X_teste, y_teste

def dataaugmentationASCUS(imagens_treinamento, imagens_teste, n_classes, pasta_classe):
    #6 classes -> cada imagem vira +1
    #3 classes -> cada imagem viram +3, 130 imagens viram +1
    #2 classes -> cada imagem vira +1
    n_imagens = len(imagens_treinamento)
    X_treinamento = []
    y_treinamento = []
    X_teste = []
    y_teste = []
    if(n_classes == 6):
        classe = 1 #ASCUS
    elif(n_classes == 2):
        classe = 0
    else:
        classe = 1
    for x in range(0,n_imagens):
        img = cv2.imread(pasta_classe + imagens_treinamento[x])
        X_treinamento.append(img)
        y_treinamento.append(classe)
    if (n_classes == 3):
        imagens_operacao = random.sample(range(0,n_imagens), n_imagens)
        for pos in range(0,len(imagens_operacao)):
            if(pos < 130):
                operacoes = random.sample(range(1,10), 4)
            else:
                operacoes = random.sample(range(1,10), 3)
            for operacao in operacoes:
                nova_img = operacaoAugmentation_retornar(operacao, X_treinamento[pos])
                X_treinamento.append(nova_img)
                y_treinamento.append(classe)
    if (n_classes == 6 or n_classes == 2):
        for x in range(0,n_imagens):
            operacoes = random.sample(range(1,10), 2)
            for operacao in operacoes:
                nova_img = operacaoAugmentation_retornar(operacao, X_treinamento[x])
                X_treinamento.append(nova_img)
                y_treinamento.append(classe)
    for x in range(0,len(imagens_teste)):
        X_teste.append(cv2.imread(pasta_classe + imagens_teste[x]))
        y_teste.append(classe)
    return X_treinamento, y_treinamento, X_teste, y_teste
    
def dataaugmentationHSIL(imagens_treinamento, imagens_teste, n_classes, pasta_classe):
    #6 classes -> sem balancear
    #3 classes -> sem balancear
    #2 classes -> sem balancear
    n_imagens = len(imagens_treinamento)
    X_treinamento = []
    y_treinamento = []
    X_teste = []
    y_teste = []
    if(n_classes == 6):
        classe = 3 #HSIL
    else:
        classe = 0
    for x in range(0,n_imagens):
        img = cv2.imread(pasta_classe + imagens_treinamento[x])
        X_treinamento.append(img)
        y_treinamento.append(classe)
    for x in range(0,len(imagens_teste)):
        X_teste.append(cv2.imread(pasta_classe + imagens_teste[x]))
        y_teste.append(classe)
    return X_treinamento, y_treinamento, X_teste, y_teste

def dataaugmentationASCH(imagens_treinamento, imagens_teste, n_classes, pasta_classe):
    #6 classes -> 270 imagens viram +1
    #3 classes -> 270 imagens viram +1
    #2 classes -> 270 imagens viram +1
    n_imagens = len(imagens_treinamento)
    X_treinamento = []
    y_treinamento = []
    X_teste = []
    y_teste = []
    classe = 0 #ASCH
    for x in range(0,n_imagens):
        img = cv2.imread(pasta_classe + imagens_treinamento[x])
        X_treinamento.append(img)
        y_treinamento.append(classe)
    imagens_operacao = random.sample(range(0,n_imagens), 270)
    for imgs_op in imagens_operacao:
        operacao = random.randint(1,10)
        nova_img = operacaoAugmentation_retornar(operacao, X_treinamento[imgs_op])
        X_treinamento.append(nova_img)
        y_treinamento.append(classe)
    for x in range(0,len(imagens_teste)):
        X_teste.append(cv2.imread(pasta_classe + imagens_teste[x]))
        y_teste.append(classe)
    return X_treinamento, y_treinamento, X_teste, y_teste
    
def dataaugmentationCA(imagens_treinamento, imagens_teste, n_classes, pasta_classe):
    #6 classes -> cada CA vira +10
    #3 classes -> cada CA vira +10
    #2 classes -> cada CA vira +10
    n_imagens = len(imagens_treinamento)
    X_treinamento = []
    y_treinamento = []
    X_teste = []
    y_teste = []
    if(n_classes == 6):
        classe = 2 #CA
    else:
        classe = 0
    for x in range(0,n_imagens):
        img = cv2.imread(pasta_classe + imagens_treinamento[x])
        X_treinamento.append(img)
        y_treinamento.append(classe)
        for operacao in range(1,11):
            nova_img = operacaoAugmentation_retornar(operacao, img)
            X_treinamento.append(nova_img)
            y_treinamento.append(classe)
    for x in range(0,len(imagens_teste)):
        X_teste.append(cv2.imread(pasta_classe + imagens_teste[x]))
        y_teste.append(classe)
    return X_treinamento, y_treinamento, X_teste, y_teste
    

def dataaugmentationNormal(imagens_treinamento, imagens_teste, n_classes, pasta_classe):
    #6 classes -> ja balanceado
    #3 classes -> cada normal vira +2
    #2 classes -> cada imagem vira +4
    n_imagens = len(imagens_treinamento)
    X_treinamento = []
    y_treinamento = []
    X_teste = []
    y_teste = []
    if(n_classes == 6):
        classe = 5 #Normal
    elif(n_classes == 3):
        classe = 2
    else:
        classe = 1
    if(n_classes == 6):
    #sem balancear
        for x in range(0,n_imagens):
            img = cv2.imread(pasta_classe + imagens_treinamento[x])
            X_treinamento.append(img)
            y_treinamento.append(classe)
    if(n_classes == 3):
    # cada imagem vira +2
        for x in range(0,n_imagens):
            img = cv2.imread(pasta_classe + imagens_treinamento[x])
            X_treinamento.append(img)
            y_treinamento.append(classe)
            operacoes = random.sample(range(1,10), 4)
            for operacao in operacoes:
                nova_img = operacaoAugmentation_retornar(operacao, img)
                X_treinamento.append(nova_img)
                y_treinamento.append(classe)
    if(n_classes == 2):
    #cada imagem vira +4
        for x in range(0,n_imagens):
            img = cv2.imread(pasta_classe + imagens_treinamento[x])
            X_treinamento.append(img)
            y_treinamento.append(classe)
            operacoes = random.sample(range(1,10), 4)
            for operacao in operacoes:
                nova_img = operacaoAugmentation_retornar(operacao, img)
                X_treinamento.append(nova_img)
                y_treinamento.append(classe)
    for x in range(0,len(imagens_teste)):
        X_teste.append(cv2.imread(pasta_classe + imagens_teste[x]))
        y_teste.append(classe)
    return X_treinamento, y_treinamento, X_teste, y_teste
            
def operacaoAugmentation_retornar(operacao, img):
    # rotacionar
    if(operacao == 1):
        nova_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif(operacao == 2):
        nova_img = cv2.rotate(img, cv2.ROTATE_180)
    elif(operacao == 3):
        nova_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # espelhar
    elif(operacao == 4):
        nova_img= cv2.flip(img, 1)
    elif(operacao == 5):
        img_rotate_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        nova_img = cv2.flip(img_rotate_90, 1)
    elif(operacao == 6):
        img_rotate_180 = cv2.rotate(img, cv2.ROTATE_180)
        nova_img = cv2.flip(img_rotate_180, 1)
    elif(operacao == 7):
        img_rotate_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        nova_img = cv2.flip(img_rotate_270, 1)
    elif(operacao == 8):
        sigma = 0.05 
        noisy = random_noise(img, var=sigma**2)
        nova_img = noisy
        nova_img = nova_img * 255
    elif(operacao == 9):
        sigma = 0.005 
        noisy = random_noise(img, var=sigma**2)
        nova_img = denoise_tv_chambolle(noisy, weight=0.05, multichannel=True)
        nova_img = nova_img * 255
    elif(operacao == 10):
        sigma = 0.005 
        noisy = random_noise(img, var=sigma**2)
        nova_img = denoise_bilateral(noisy, sigma_color=0.01, sigma_spatial=5, multichannel=True)
        nova_img = nova_img * 255
    return nova_img

def salvar_BalanceamentoDividido(num_classes):
    X_treinamento, y_treinamento, X_validacao, y_validacao, X_teste, y_teste = dividirEBalancearPorClasse(num_classes, 0.2)
    pasta_salvar = '..\\Base balanceada dividida\\' + str(num_classes) + ' classes\\'
    for x in range(0,len(X_treinamento)):
        novo_nome = pasta_salvar + "\\Treino\\" + str(x) + "_" + str(y_treinamento[x]) + "_" + ".png"
        cv2.imwrite(novo_nome, X_treinamento[x])
    for x in range(0,len(X_validacao)):
        novo_nome = pasta_salvar + "\\Validacao\\" + str(x) + "_" + str(y_validacao[x]) + "_" + ".png"
        cv2.imwrite(novo_nome, X_validacao[x])
    for x in range(0,len(X_teste)):
        novo_nome = pasta_salvar + "\\Teste\\" + str(x) + "_" + str(y_teste[x]) + "_" + ".png"
        cv2.imwrite(novo_nome, X_teste[x])
        
def ler_BalanceamentoDividido(num_classes):
    pasta_ler = '..\\Base balanceada dividida\\' + str(num_classes) + ' classes\\'
    X_treinamento, y_treinamento, X_validacao, y_validacao, X_teste, y_teste = [], [], [], [], [], []
    for arq in os.listdir(pasta_ler + '\\Treino\\'):
        X_treinamento.append(cv2.imread(pasta_ler + '\\Treino\\' + arq))
        y_treinamento.append(arq.split('_')[1])
    for arq in os.listdir(pasta_ler + '\\Validacao\\'):
        X_validacao.append(cv2.imread(pasta_ler + '\\Validacao\\' + arq))
        y_validacao.append(arq.split('_')[1])
    for arq in os.listdir(pasta_ler + '\\Teste\\'):
        X_teste.append(cv2.imread(pasta_ler + '\\Teste\\' + arq))
        y_teste.append(arq.split('_')[1])
    return X_treinamento, y_treinamento, X_validacao, y_validacao, X_teste, y_teste
