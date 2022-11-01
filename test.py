from metricas import dice_arteria_derecha, dice_arteria_izquierda, dice_arteria_principal, dice_coefficient, dice_promedio
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

def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
fix_gpu()


def rediminesionar(self,volumen):
    x = np.linspace(0,volumen.shape[0]-1,volumen.shape[0]) 
    y = np.linspace(0,volumen.shape[1]-1,volumen.shape[1]) 
    z = np.linspace(0,volumen.shape[2]-1,volumen.shape[2]) 

    f = RegularGridInterpolator((x,y,z), volumen)
    xn = np.linspace(0,volumen.shape[0]-1,self.dim[0])
    yn = np.linspace(0,volumen.shape[1]-1,self.dim[1]) 
    zn = np.linspace(0,volumen.shape[2]-1,self.dim[2]) 

    new_grid = np.array(np.meshgrid(xn,yn,zn, indexing = 'ij'))
    new_grid = np.moveaxis(new_grid, 0, -1)  #ordena ejes 
    data_new = f(new_grid)

    return data_new


def test_modelo(ruta_modelo,conjuntos, imagenes, etiquetas):

    model = load_model(ruta_modelo,custom_objects={"dice_coefficient_loss":dice_coefficient_loss,
                                                    "dice_promedio": dice_promedio,
                                                    "dice_arteria_principal":dice_arteria_principal,
                                                    "dice_arteria_izquierda":dice_arteria_izquierda,
                                                    "dice_arteria_derecha":dice_arteria_derecha
                                                    })
    for id in range(0,len(conjuntos)):

        ruta_imagen = imagenes[id] 
        ruta_etiqueta = etiquetas[id]

        imagen, _ = nrrd.read(ruta_imagen)
        imagen = rediminesionar(imagen)
        imagen = np.expand_dims(imagen, axis=3)


        etiqueta, _ = nrrd.read(ruta_etiqueta)

        etiqueta = utils.to_categorical(etiqueta, num_classes=self.clases)
        etiqueta = rediminesionar(etiqueta)
        etiqueta = (etiqueta > 0).astype(int)
        print("etiqueta shape: ",etiqueta.shape)
        print("etiqueta unique: ",np.unique(etiqueta))

        output = model.predict(
                        imagen,
                        verbose="auto",
                        workers=1,
                    )

        print("output: {}".format(output.shape))







    




    #print("x.shape: ",x.shape)
    #results = model.evaluate(generador_test,return_dict=True)
    #print(results)
    #print(model.metrics_names)



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

    model = build_unet(dim, n_classes=4)

    optimizer = optimizers.Adam(0.01) 
    model.compile(optimizer = optimizer, loss=dice_coefficient_loss, metrics=[dice_promedio,dice_arteria_principal,dice_arteria_izquierda,dice_arteria_derecha],run_eagerly=True)
    model.save("peso_test.h5")

    test_modelo("peso_test.h5",conjuntos, imagenes, etiquetas)
