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

from metricas import dice_arteria_derecha
from matplotlib import pyplot as plt

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


def test_modelo(ruta_modelo,conjuntos, imagenes, etiquetas):

    model = load_model(ruta_modelo,custom_objects={"dice_coefficient_loss":dice_coefficient_loss,
                                                    "dice_promedio": dice_promedio,
                                                    "dice_arteria_principal":dice_arteria_principal,
                                                    "dice_arteria_izquierda":dice_arteria_izquierda,
                                                    "dice_arteria_derecha":dice_arteria_derecha})
    detecciones = []
    true_etiqueta = []
    inputs = []
    for id in conjuntos:
        ruta_imagen = imagenes[id] 
        ruta_etiqueta = etiquetas[id]

        imagen, _ = nrrd.read(ruta_imagen)
        imagen = rediminesionar(imagen)

        dims =  imagen.shape
        imagen = np.expand_dims(imagen, axis=3)
        imagen = np.expand_dims(imagen, axis=0)


        etiqueta, _ = nrrd.read(ruta_etiqueta)

        plt.imshow(etiqueta[:,:,32])
        plt.savefig("test0.png")

        etiqueta = utils.to_categorical(etiqueta, num_classes=4)
        etiqueta = rediminesionar(etiqueta)

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
    print(np.unique(detecciones[0][:,:,32,2]))

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
    ax[4,1].imshow(detecciones[4][:,:,32,2]>0.1, cmap="gray")
    ax[4,2].imshow(detecciones[4][:,:,32,2]>0.3, cmap="gray")
    ax[4,3].imshow(detecciones[4][:,:,32,2]>0.5, cmap="gray")
    ax[4,4].imshow(detecciones[4][:,:,32,2]>0.7, cmap="gray")
    ax[4,5].imshow(detecciones[4][:,:,32,2]>0.9, cmap="gray")
    ax[4,6].imshow(true_etiqueta[4][:,:,32,1])

    plt.savefig("test2.png")


if __name__ == "__main__":

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
