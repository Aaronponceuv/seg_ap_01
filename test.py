from metricas import dice_arteria_derecha, dice_arteria_izquierda, dice_arteria_principal, dice_coefficient, dice_promedio,dice_background
from scipy.interpolate import RegularGridInterpolator
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
from perdidas import dice_coefficient_loss
from keras.models import load_model
from dataloader import DataLoader
from modelo import build_unet
from  keras import optimizers
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


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
fix_gpu()


def rediminesionar(volumen):
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


def test_modelo(ruta_modelo,conjuntos, imagenes, etiquetas):

    model = load_model(ruta_modelo,custom_objects={"dice_coefficient_loss":dice_coefficient_loss,
                                                    "dice_promedio": dice_promedio,
                                                    "dice_arteria_principal":dice_arteria_principal,
                                                    "dice_arteria_izquierda":dice_arteria_izquierda,
                                                    "dice_arteria_derecha":dice_arteria_derecha})
    detecciones = []
    true_etiqueta = []
    inputs = []
    true_etiquetas_no_binarias = []
    #redim_m = Resized(keys=["image", "seg","seg_art_principal","seg_art_derecha","seg_art_izquierda"], spatial_size=(128,128,64), mode=('nearest'))
    #data_dict = redim_m(data_dict)
    for id in conjuntos:
        ruta_imagen = imagenes[id] 
        ruta_etiqueta = etiquetas[id]

        imagen, _ = nrrd.read(ruta_imagen)
        imagen = rediminesionar(imagen)

        dims =  imagen.shape
        imagen = np.expand_dims(imagen, axis=3)
        imagen = np.expand_dims(imagen, axis=0)


        etiqueta_no_binaria, _ = nrrd.read(ruta_etiqueta)
        true_etiquetas_no_binarias.append(etiqueta_no_binaria)

        arteria_principal = etiqueta_no_binaria == 1
        arteria_derecha = etiqueta_no_binaria == 2
        arteria_izquierda = etiqueta_no_binaria == 3


        arteria_principal = np.expand_dims(arteria_principal, axis=0)
        arteria_derecha = np.expand_dims(arteria_derecha, axis=0)
        arteria_izquierda = np.expand_dims(arteria_izquierda, axis=0)

        #Resized(arteria_izquierda,mode="nearest")


        plt.imshow(etiqueta_no_binaria[:,:,32])
        plt.savefig("test0.png")

        etiqueta = utils.to_categorical(etiqueta_no_binaria, num_classes=4)
        etiqueta = rediminesionar(etiqueta)

        print("etiqueta redim : ",etiqueta.shape)

        plt.imshow(etiqueta[:,:,32])
        plt.savefig("test1.png")

        

        etiqueta = (etiqueta > 0).astype(int)

        output = model.predict(
                        imagen,
                        verbose="auto",
                        workers=1,
                    )

        etiqueta = np.expand_dims(etiqueta, axis=0)

        dice_derecha = dice_arteria_derecha(etiqueta.astype(np.float32),output)
        dice_principal = dice_arteria_principal(etiqueta.astype(np.float32),output)
        dice_izquierda = dice_arteria_izquierda(etiqueta.astype(np.float32),output)
        dice_back = dice_background(etiqueta.astype(np.float32),output)
        dice_prom = dice_promedio(etiqueta.astype(np.float32),output)

        print("-----Resumen Test IMG: {}:------\n - . Dice Principal: {} \n - . Dice Derecha: {}\n - . Dice Izquierdo: {}\n - . Dice Promedio: {}\n - . Dice background: {} ".format(
                                            id,dice_principal.numpy(),dice_derecha.numpy(),dice_izquierda.numpy(),dice_prom.numpy(),dice_back.numpy()))
        
        inputs.append(np.array(imagen[0]).reshape(dims))
        detecciones.append(np.array(output[0]).reshape(128,128,64,4))
        true_etiqueta.append(np.array(etiqueta[0]).reshape(128,128,64,4))

    print("true_etiqueta: {}".format(true_etiqueta[0].shape))
    print(np.max(detecciones[0][:,:,32,2]))



    fig, ax = plt.subplots(5,7, figsize=(10,10))
    ax[0,0].imshow(inputs[0][:,:,32], cmap="gray")
    ax[0,1].imshow(detecciones[0][:,:,32,2]>0.1, cmap="gray")
    ax[0,2].imshow(detecciones[0][:,:,32,2]>0.3, cmap="gray")
    ax[0,3].imshow(detecciones[0][:,:,32,2]>0.5, cmap="gray")
    ax[0,4].imshow(detecciones[0][:,:,32,2]>0.7, cmap="gray")
    ax[0,5].imshow(detecciones[0][:,:,32,2]>0.9, cmap="gray")
    ax[0,6].imshow(true_etiqueta[0][:,:,32,1])

    ax[1,0].imshow(inputs[1][:,:,32], cmap="gray")
    ax[1,1].imshow(detecciones[1][:,:,32,2]>0.1, cmap="gray")
    ax[1,2].imshow(detecciones[1][:,:,32,2]>0.3, cmap="gray")
    ax[1,3].imshow(detecciones[1][:,:,32,2]>0.5, cmap="gray")
    ax[1,4].imshow(detecciones[1][:,:,32,2]>0.7, cmap="gray")
    ax[1,5].imshow(detecciones[1][:,:,32,2]>0.9, cmap="gray")
    ax[1,6].imshow(true_etiqueta[1][:,:,32,1])

    ax[2,0].imshow(inputs[2][:,:,32], cmap="gray")
    ax[2,1].imshow(detecciones[2][:,:,32,2]>0.1, cmap="gray")
    ax[2,2].imshow(detecciones[2][:,:,32,2]>0.3, cmap="gray")
    ax[2,3].imshow(detecciones[2][:,:,32,2]>0.5, cmap="gray")
    ax[2,4].imshow(detecciones[2][:,:,32,2]>0.7, cmap="gray")
    ax[2,5].imshow(detecciones[2][:,:,32,2]>0.9, cmap="gray")
    ax[2,6].imshow(true_etiqueta[2][:,:,32,1])

    ax[3,0].imshow(inputs[3][:,:,32], cmap="gray")
    ax[3,1].imshow(detecciones[3][:,:,32,2]>0.1, cmap="gray")
    ax[3,2].imshow(detecciones[3][:,:,32,2]>0.3, cmap="gray")
    ax[3,3].imshow(detecciones[3][:,:,32,2]>0.5, cmap="gray")
    ax[3,4].imshow(detecciones[3][:,:,32,2]>0.7, cmap="gray")
    ax[3,5].imshow(detecciones[3][:,:,32,2]>0.9, cmap="gray")
    ax[3,6].imshow(true_etiqueta[3][:,:,32,1])
                                                                                                                                                                                 
    ax[4,0].imshow(inputs[4][:,:,32], cmap="gray")
    ax[4,1].imshow((detecciones[4][:,:,32,2]>0.1).astype(np.int8), cmap="gray")
    ax[4,2].imshow((detecciones[4][:,:,32,2]>0.3).astype(np.int8), cmap="gray")
    ax[4,3].imshow((detecciones[4][:,:,32,2]>0.5).astype(np.int8), cmap="gray")
    ax[4,4].imshow((detecciones[4][:,:,32,2]>0.7).astype(np.int8), cmap="gray")
    ax[4,5].imshow((detecciones[4][:,:,32,2]>0.9).astype(np.int8), cmap="gray")
    ax[4,6].imshow(true_etiqueta[4][:,:,32,1])

    plt.savefig("test2.png")

    print("nop unique: ",np.unique((detecciones[4][:,:,32,2]>0.5).astype(np.int8)))
    print("detecciones[0][:,:,:,:]: ",detecciones[0][:,:,:,:].shape)

    print("np unique etiqueta ", np.unique(true_etiquetas_no_binarias[0][:,32,:]))

    plt.clf()
    plt.close()

    plt.imshow(true_etiquetas_no_binarias[0][:,32,:])
    plt.savefig("test_true.png")

    mask_data = np.array([[1, 2, 2, ... , 2, 2, 1], ...])

    class_labels = {
        0: "Background",
        1: "Art. pulmonar principal",
        2: "Art. pulmonar derecha",
        3: "Art. pulmonar izquierda"
    }



    img = (inputs[0][:,:,32]).astype(np.float16)


    mask_img = wandb.Image(img, masks={
                "predictions": {
                    "mask_data": detecciones[0][:,:,32,2],
                    "class_labels": class_labels
                },
                "ground_truth" : {
                        "mask_data" : true_etiquetas_no_binarias[0][:,32,:],
                        "class_labels" : class_labels
                    }

            })

    wandb.log({"test":mask_img})


    img = (inputs[0][:,:,32]).astype(np.int8) * 255
    mask_img = wandb.Image(img, masks={
                "predictions": {
                    "mask_data": detecciones[0][:,32,2],
                    "class_labels": class_labels
                },
                "ground_truth" : {
                        "mask_data" : true_etiquetas_no_binarias[0][:,32,:],
                        "class_labels" : class_labels
                    }

            })

    wandb.log({"test":mask_img})

    img = (inputs[0][:,:,32]).astype(np.int8) * 255
    mask_img = wandb.Image(img, masks={
                "predictions": {
                    "mask_data": detecciones[0][:,32,2],
                    "class_labels": class_labels
                },
                "ground_truth" : {
                        "mask_data" : (true_etiquetas_no_binarias[0][:,32,:]),
                        "class_labels" : class_labels
                    }

            })

    wandb.log({"test":mask_img})

    img = (inputs[0][:,:,32]).astype(np.int8) * 255
    mask_img = wandb.Image(img, masks={
                "predictions": {
                    "mask_data": detecciones[0][:,:,32,2],
                    "class_labels": class_labels
                },
                "ground_truth" : {
                        "mask_data" : true_etiquetas_no_binarias[0][:,32,:],
                        "class_labels" : class_labels
                    }

            })

    wandb.log({"test":mask_img})


    #recorrer(detecciones[4][:,:,32,2]>0.3)


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
    generador_test = DataLoader(conjuntos["test"], imagenes, etiquetas, 
                                        batch_size=1,
                                        dim=dim[0:3],
                                        shuffle=False,
                                        n_clases=4,
                                        canales=1,
                                        tipo_de_dato=".nrrd")

    #model = build_unet(dim, n_classes=4)
#
    #optimizer = optimizers.Adam(0.01) 
    #model.compile(optimizer = optimizer, loss=dice_coefficient_loss, metrics=[dice_promedio,dice_arteria_principal,dice_arteria_izquierda,dice_arteria_derecha],run_eagerly=True)
    #model.save("peso_test.h5")

    test_modelo("peso_test.h5",conjuntos["test"], imagenes, etiquetas)
