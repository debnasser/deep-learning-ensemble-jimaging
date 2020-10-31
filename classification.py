import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Recall
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNet, Xception, InceptionV3
import efficientnet.tfkeras as efn 
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import multilabel_confusion_matrix
from dataAugmentation import ler_BalanceamentoDividido

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
    print(mcm)
    tn = np.mean(mcm[:,0,0])
    tp = np.mean(mcm[:,1,1])
    fn = np.mean(mcm[:,1,0])
    fp = np.mean(mcm[:,0,1])
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

num_classes = 2
num_epocas = 1
num_batch_size = 32
X_treinamento, y_treinamento, X_validacao, y_validacao, X_teste, y_teste = ler_BalanceamentoDividido(num_classes)

X_treinamento = np.array(X_treinamento, dtype=np.int32)
X_validacao = np.array(X_validacao, dtype=np.int32)
X_teste = np.array(X_teste, dtype=np.int32)
y_treinamento = to_categorical(y_treinamento, num_classes, dtype=np.int32)
y_validacao = to_categorical(y_validacao, num_classes, dtype=np.int32)
y_teste = to_categorical(y_teste, num_classes, dtype=np.int32)

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
classificador = construirClassificador(num_classes, 1)
checkpoint = ModelCheckpoint('best_model.hdf5', monitor='val_recall', verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_recall', factor=0.5, patience=5, verbose=1)
history = classificador.fit(X_treinamento, y_treinamento, epochs=num_epocas, batch_size=num_batch_size, validation_data=(X_validacao, y_validacao), callbacks=[checkpoint])

print("\n ----- Resultados ------")
classificador.load_weights('best_model.hdf5')
y_pred = classificador.predict(X_teste)
mcm = multilabel_confusion_matrix(np.argmax(y_teste, axis=1), np.argmax(y_pred, axis=1))
imprimirResultado(mcm)
