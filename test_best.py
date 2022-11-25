from metricas import dice_arteria_derecha, dice_arteria_izquierda, dice_arteria_principal, dice_coefficient, dice_promedio,dice_background
from scipy.interpolate import RegularGridInterpolator
#from tensorflow.compat.v1 import InteractiveSession
#from tensorflow.compat.v1 import ConfigProto
from perdidas import dice_coefficient_loss
from keras.models import load_model
from dataloader import DataLoader
from keras import utils
import numpy as np
import json
import nrrd
import keras
from metricas import dice_arteria_derecha
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap, ListedColormap

import wandb
from monai.transforms import Resized
import time


#def fix_gpu():
#    config = ConfigProto()
#    config.gpu_options.allow_growth = True
#    session = InteractiveSession(config=config)
#fix_gpu()


def rediminesionar(volumen, dim):
    x = np.linspace(0,volumen.shape[0]-1,volumen.shape[0]) 
    y = np.linspace(0,volumen.shape[1]-1,volumen.shape[1]) 
    z = np.linspace(0,volumen.shape[2]-1,volumen.shape[2]) 

    f = RegularGridInterpolator((x,y,z), volumen)
    xn = np.linspace(0,volumen.shape[0]-1,dim[0])
    yn = np.linspace(0,volumen.shape[1]-1,dim[1]) 
    zn = np.linspace(0,volumen.shape[2]-1,dim[2]) 

    new_grid = np.array(np.meshgrid(xn,yn,zn, indexing = 'ij'))
    new_grid = np.moveaxis(new_grid, 0, -1)  #ordena ejes 
    data_new = f(new_grid)

    return data_new

def recorrer(segmentacion,map=None):
    segmentacion = np.asarray(segmentacion)
    segmentacion = segmentacion[~(segmentacion==0).all((2,1))]
    seg = segmentacion
    import math
    columna = 0
    fila = 0
    fig, ax = plt.subplots(4,int(math.ceil(seg.shape[0] / 4)),figsize=(20,10))
    map=ListedColormap(["indigo", "yellow", "green", "green"])
    for indice in range(0,seg.shape[0]):
 
        ax[fila,columna].imshow(seg[indice,:,:],cmap=map)
        ax[fila,columna].set_title("fase {}".format(str(indice)))
        #ax[fila,columna].axis(False)
            
        if(columna == int(math.ceil(seg.shape[0] / 4)) -1):
            columna = 0
            fila+=1
        else:
            columna +=1
        if(fila==5 and columna == 4):
            break
    plt.show()


def registrar_evaluacion(inputs,true_etiqueta,detecciones,nombre_imagenes,class_labels):
    for img,y_true,y_pred,nombre_imagen in zip(inputs,true_etiqueta,detecciones,nombre_imagenes):
        print("=> Registrando {}".format(nombre_imagen))
        #print("img {},y_true {},y_pred :{}".format(img.shape,y_true.shape,y_pred.shape))

        # Sagital
        for id in range(0,img.shape[2]):
            #print("id ",id)
            #print(img[:,:,id].shape)
            imagen = img[:,:,id] * 255

            plt.imshow(imagen, cmap="gray")
            plt.axis('off')
            plt.imsave("image.png",imagen,cmap="gray")
            time.sleep(0.2)

            #print(y_true[:,:,id,1].shape)
            #print(y_pred[:,:,id,1].shape)


            ground_truth = y_true[:,:,id,1] +y_true[:,:,id,2]*2 + y_true[:,:,id,3]*3 #+ y_true[:,:,id,4]
            predictions = y_pred[:,:,id,1] + y_pred[:,:,id,2]*2 + y_pred[:,:,id,3]*3 #+ y_pred[:,:,id,4]


            #print("np unique ground_truth: ",np.unique(ground_truth))
            #print("np unique predictions: ",np.unique(predictions))

            mask_img = wandb.Image("image.png", masks={
                "predictions": {
                    "mask_data": predictions,
                    "class_labels": class_labels
                },
                "ground_truth" : {
                        "mask_data" : ground_truth,
                        "class_labels" : class_labels
                    }
                })
            
            wandb.log({nombre_imagen+"_sagital":mask_img})


        # Coronal
        #"""
        for id in range(0,img.shape[1]):
            ##print("id ",id)
            imagen = img[:,id,:] * 255

            plt.imshow(imagen, cmap="gray")
            plt.axis('off')
            plt.imsave("image.png",imagen,cmap="gray")
            time.sleep(0.2)

            #print(y_true[:,id,:,1].shape)
            #print(y_pred[:,id,:,1].shape)


            ground_truth = y_true[:,id,:,1] +y_true[:,id,:,2]*2 + y_true[:,id,:,3]*3 #+ y_true[:,id,:,4]
            predictions = y_pred[:,id,:,1] + y_pred[:,id,:,2]*2 + y_pred[:,id,:,3]*3 #+ y_pred[:,id,:,4]


            print("clases ground_truth: ",np.unique(ground_truth))
            print("clases predictions: ",np.unique(predictions))

            mask_img = wandb.Image("image.png", masks={
                "predictions": {
                    "mask_data": predictions,
                    "class_labels": class_labels
                },
                "ground_truth" : {
                        "mask_data" : ground_truth,
                        "class_labels" : class_labels
                    }
                })
            
            wandb.log({nombre_imagen+"_coronal":mask_img})


        for id in range(0,img.shape[0]):
            #print("=] id ",id)
            imagen = img[id,:,:] * 255

            plt.imshow(imagen, cmap="gray")
            plt.axis('off')
            plt.imsave("image.png",imagen,cmap="gray")
            time.sleep(0.2)

            #print(y_true[id,:,:,1].shape)
            #print(y_pred[id,:,:,1].shape)


            ground_truth = y_true[id,:,:,1] +y_true[id,:,:,2]*2 + y_true[id,:,:,3]*3 #+ y_true[id,:,:,4]
            predictions = y_pred[id,:,:,1] + y_pred[id,:,:,2]*2 + y_pred[id,:,:,3]*3 #+ y_pred[id,:,:,4]


            print("clases ground_truth: ",np.unique(ground_truth))
            print("clases predictions: ",np.unique(predictions))

            mask_img = wandb.Image("image.png", masks={
                "predictions": {
                    "mask_data": predictions,
                    "class_labels": class_labels
                },
                "ground_truth" : {
                        "mask_data" : ground_truth,
                        "class_labels" : class_labels
                    }
                })
            
            wandb.log({nombre_imagen+"_axial":mask_img})


def evaluar_imagenes(model,conjuntos,imagenes,etiquetas,dimension_target):
    detecciones = []
    true_etiqueta = []
    inputs = []
    true_etiquetas_no_binarias = []

    for id in conjuntos:
        ruta_imagen = imagenes[id] 
        ruta_etiqueta = etiquetas[id]

        imagen, _ = nrrd.read(ruta_imagen)
        imagen = rediminesionar(imagen,dimension_target)

        dims =  imagen.shape
        


        etiqueta_no_binaria, _ = nrrd.read(ruta_etiqueta)
        true_etiquetas_no_binarias.append(etiqueta_no_binaria)


        etiqueta = utils.to_categorical(etiqueta_no_binaria, num_classes=4)
        etiqueta = rediminesionar(etiqueta,dimension_target)
        etiqueta = (etiqueta > 0.5).astype(int)
        #plt.imshow(etiqueta[:,:,20])
        #plt.savefig("test1.png")
        #plt.close()

        #print("etiqueta shape: {}".format(etiqueta.shape))
        #print("imagen shape: {}".format(imagen.shape))



        imagen = np.expand_dims(imagen, axis=3)
        imagen = np.expand_dims(imagen, axis=0)

        etiqueta = np.expand_dims(etiqueta, axis=0)

        output = model.predict(
                        imagen,
                        verbose="auto",
                        workers=1,
                    )
        #debug
        #output = np.random.randint(2, size=(1, 128, 128, 64, 4)).astype(np.float32)
        #print("Etiqueta unique: {} shape:{} ".format(np.unique(etiqueta.astype(np.float32)), etiqueta.shape))

        output = (output > 0.5).astype(np.float32)

        etiqueta = etiqueta.astype(np.float32)

        dice_derecha = dice_arteria_derecha(etiqueta,output)
        dice_principal = dice_arteria_principal(etiqueta,output)
        dice_izquierda = dice_arteria_izquierda(etiqueta,output)
        dice_back = dice_background(etiqueta,output)
        dice_prom = dice_promedio(etiqueta,output)

        print("-----Resumen Test IMG: {}:------\n - . Dice Principal: {} \n - . Dice Derecha: {}\n - . Dice Izquierdo: {}\n - . Dice Promedio: {}\n - . Dice background: {} ".format(
                                            id,dice_principal.numpy(),dice_derecha.numpy(),dice_izquierda.numpy(),dice_prom.numpy(),dice_back.numpy()))

        inputs.append(np.array(imagen[0]).reshape(dims))
        detecciones.append(np.array(output[0]).reshape(128,128,64,4))
        true_etiqueta.append(np.array(etiqueta[0]).reshape(128,128,64,4))

    return inputs, detecciones, true_etiqueta






def test_modelo(ruta_modelo,conjuntos, imagenes, etiquetas, dimensiones_imagen):
    print("--> En Fase de Test")
    class_labels = {
        0: "Background",
        1: "Art. pulmonar principal",
        2: "Art. pulmonar derecha",
        3: "Art. pulmonar izquierda"
    }

    nombre_imagenes = list(conjuntos)

    model = load_model(ruta_modelo,custom_objects={"dice_coefficient_loss":dice_coefficient_loss,
                                                    "dice_promedio": dice_promedio,
                                                    "dice_arteria_principal":dice_arteria_principal,
                                                    "dice_arteria_izquierda":dice_arteria_izquierda,
                                                    "dice_arteria_derecha":dice_arteria_derecha})
    

    
    inputs, detecciones, true_etiqueta = evaluar_imagenes(model,conjuntos,imagenes,etiquetas,dimensiones_imagen)
    registrar_evaluacion(inputs,true_etiqueta,detecciones,nombre_imagenes,class_labels)



   

if __name__ == "__main__":

    wandb.init(settings=wandb.Settings(start_method="fork"))

    dim=(128, 128, 64,1)

    directorio_rutas = "../rutas_de_dataset/v2"

    conjuntos = directorio_rutas + "/ids.json"
    etiquetas = directorio_rutas + "/etiquetas.json"
    imagenes  = directorio_rutas + "/imagenes.json"

    conjuntos = open(conjuntos)
    etiquetas = open(etiquetas)
    imagenes = open(imagenes)

    conjuntos = json.load(conjuntos)
    etiquetas = json.load(etiquetas)
    imagenes = json.load(imagenes)
    
    print(conjuntos["test"])


    test_modelo("peso_test.h5",conjuntos["test"], imagenes, etiquetas)
