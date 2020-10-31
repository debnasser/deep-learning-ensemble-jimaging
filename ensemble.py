import efficientnet.tfkeras as efn 
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.metrics import Recall
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, BatchNormalization
from sklearn.ensemble import VotingClassifier
from dataAugmentation import dividirEBalancearPorClasse, ler_BalanceamentoDividido
import numpy as np
from tensorflow.keras.applications import MobileNetV2, MobileNet, NASNetMobile, DenseNet121, DenseNet201, Xception, InceptionV3
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from skmultilearn.ensemble import voting
from tensorflow import math
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def construirClassificador(num_classes, num_classificador):
    # 1 - EfficientNetB0
    # 2 - EfficientNetB1
    # 3 - EfficientNetB2
    # 4 - EfficientNetB3
    # 5 - EfficientNetB4
    # 6 - EfficientNetB5
    # 7 - EfficientNetB6
    # 8 - MobileNet
    # 9 - Xception
    # 10 - InceptionV3
    if (num_classificador == 1):
        base_model = efn.EfficientNetB0(include_top=False, input_shape=(90,90,3), classes= num_classes)
    elif(num_classificador == 2):
        base_model = efn.EfficientNetB1(include_top=False, input_shape=(90,90,3), classes= num_classes)
    elif(num_classificador == 3):
        base_model = efn.EfficientNetB2(include_top=False, input_shape=(90,90,3), classes= num_classes)
    elif(num_classificador == 4):
        base_model = efn.EfficientNetB3(include_top=False, input_shape=(90,90,3), classes= num_classes)
    elif(num_classificador == 5):
        base_model = efn.EfficientNetB4(include_top=False, input_shape=(90,90,3), classes= num_classes)
    elif(num_classificador == 6):
        base_model = efn.EfficientNetB5(include_top=False, input_shape=(90,90,3), classes= num_classes)
    elif(num_classificador == 7):
        base_model = efn.EfficientNetB6(include_top=False, input_shape=(90,90,3), classes= num_classes)
    elif(num_classificador == 8):
        base_model = MobileNet(include_top=False, input_shape=(90,90,3), classes= num_classes)
    elif(num_classificador == 9):
        base_model = Xception(include_top=False, input_shape=(90,90,3), classes= num_classes)
    elif(num_classificador == 10):
        base_model = InceptionV3(include_top=False, input_shape=(90,90,3), classes= num_classes)
    x = base_model.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    classificador = Model(inputs = base_model.input, outputs = predictions)
    classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[Recall(name='recall')])
    return classificador

def imprimirResultado(mcm):
    tn = np.mean(mcm[:,0,0])
    tp = np.mean(mcm[:,1,1])
    fn = np.mean(mcm[:,0,1])
    fp = np.mean(mcm[:,1,0])
    prec = round(tp/(tp + fp), 2)
    rec = round(tp/(tp + fn), 2)
    f1 = round((2 * prec * rec) / (prec + rec), 2)
    acuracia = round((tp + tn) / (tp + tn + fp + fn), 2)
    especificidade = round(tn/ (tn + fp), 2)
    print("Precisao:", prec)
    print("Revocação:", rec)
    print("F1:", f1)
    print("Acurácia:", acuracia)
    print("Especificidade", especificidade)

num_classes = 6
X_treinamento, y_treinamento, X_validacao, y_validacao, X_teste, y_teste = ler_BalanceamentoDividido(num_classes)
X_treinamento = np.array(X_treinamento, dtype=np.int32)
X_validacao = np.array(X_validacao, dtype=np.int32)
X_teste = np.array(X_teste, dtype=np.int32)
y_treinamento = to_categorical(y_treinamento, num_classes, dtype=np.int32)
y_validacao = to_categorical(y_validacao, num_classes, dtype=np.int32)
y_teste = to_categorical(y_teste, num_classes, dtype=np.int32)

pasta = '../Modelos ' + str(num_classes) + ' classes/'

# 1 - EfficientNetB0
# 2 - EfficientNetB1
# 3 - EfficientNetB2
# 4 - EfficientNetB3
# 5 - EfficientNetB4
# 6 - EfficientNetB5
# 7 - EfficientNetB6
# 8 - MobileNet
# 9 - Xception
# 10 - InceptionV3

# Carrega os melhores modelos e pesos encontrados anteriormente para cada classe
if(num_classes == 2):
    primeiro_ef = construirClassificador(num_classes, 4)
    primeiro_ef.load_weights((pasta + 'best_model_ef3_2classes.hdf5'))
    y_pred_primeiro_ef = primeiro_ef.predict(X_teste)

    segundo_ef = construirClassificador(num_classes, 5)
    segundo_ef.load_weights((pasta + 'best_model_ef4_2classes.hdf5'))
    y_pred_segundo_ef = segundo_ef.predict(X_teste)
    
    terceiro_ef = construirClassificador(num_classes, 7)
    terceiro_ef.load_weights((pasta + 'best_model_ef6_2classes.hdf5'))
    y_pred_terceiro_ef = terceiro_ef.predict(X_teste)
    
if(num_classes == 3):
    primeiro_ef = construirClassificador(num_classes, 3)
    primeiro_ef.load_weights((pasta + 'best_model_ef2_3classes.hdf5'))
    y_pred_primeiro_ef = primeiro_ef.predict(X_teste)
    
    segundo_ef = construirClassificador(num_classes, 5)
    segundo_ef.load_weights((pasta + 'best_model_ef4_3classes.hdf5'))
    y_pred_segundo_ef = segundo_ef.predict(X_teste)
    
    terceiro_ef = construirClassificador(num_classes, 6)
    terceiro_ef.load_weights((pasta + 'best_model_ef5_3classes.hdf5'))
    y_pred_terceiro_ef = terceiro_ef.predict(X_teste)

if(num_classes == 6):
    primeiro_ef = construirClassificador(num_classes, 4)
    primeiro_ef.load_weights((pasta + 'best_model_ef3_6classes.hdf5'))
    y_pred_primeiro_ef = primeiro_ef.predict(X_teste)
    
    segundo_ef = construirClassificador(num_classes, 5)
    segundo_ef.load_weights((pasta + 'best_model_ef4_6classes.hdf5'))
    y_pred_segundo_ef = segundo_ef.predict(X_teste)
    
    terceiro_ef = construirClassificador(num_classes, 6)
    terceiro_ef.load_weights((pasta + 'best_model_ef5_6classes.hdf5'))
    y_pred_terceiro_ef = terceiro_ef.predict(X_teste)


print("\n\n ----- Resultado Ensemble -----")
y_pred_ensemble = []
y_pred_primeiro_ef = np.argmax(y_pred_primeiro_ef, axis=1)
y_pred_segundo_ef = np.argmax(y_pred_segundo_ef, axis=1)
y_pred_terceiro_ef = np.argmax(y_pred_terceiro_ef, axis=1)
for x in range(0, len(y_teste)):
    votos = np.zeros(num_classes)
    votos[y_pred_primeiro_ef[x]] = votos[y_pred_primeiro_ef[x]]+1
    votos[y_pred_segundo_ef[x]] = votos[y_pred_segundo_ef[x]]+1
    votos[y_pred_terceiro_ef[x]] = votos[y_pred_terceiro_ef[x]]+1
    if(len(np.where(votos == max(votos))[0]) == 1):
        # Por testes anteriores, o efficientnetB4 e o melhor independente do numero de classes
        y_pred_ensemble.append(np.where(votos == max(votos))[0][0])
    else:
        y_pred_ensemble.append(y_pred_segundo_ef[x])
mcm = multilabel_confusion_matrix(np.argmax(y_teste, axis=1), np.array(y_pred_ensemble))
imprimirResultado(mcm)
