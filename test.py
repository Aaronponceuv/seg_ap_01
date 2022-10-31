from keras.models import load_model
from dataloader import DataLoader
from modelo import build_unet
from metricas import dice_arteria_derecha, dice_arteria_izquierda, dice_arteria_principal, dice_coefficient, dice_promedio
from perdidas import dice_coefficient_loss
from  keras import optimizers
import json

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
fix_gpu()


def test_modelo(ruta_modelo,generador_test):
    model = load_model(ruta_modelo,custom_objects={"dice_coefficient_loss":dice_coefficient_loss,
                                                    "dice_promedio": dice_promedio,
                                                    "dice_arteria_principal":dice_arteria_principal,
                                                    "dice_arteria_izquierda":dice_arteria_izquierda,
                                                    "dice_arteria_derecha":dice_arteria_derecha
                                                    })

    results = model.evaluate(generador_test, batch_size=1)
    print(results)



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
                                        batch_size=len(conjuntos["test"]),
                                        dim=dim[0:3],
                                        shuffle=True,
                                        n_clases=4,
                                        canales=1,
                                        tipo_de_dato=".nrrd")

    model = build_unet(dim, n_classes=4)

    optimizer = optimizers.Adam(0.01) 
    model.compile(optimizer = optimizer, loss=dice_coefficient_loss, metrics=[dice_promedio,dice_arteria_principal,dice_arteria_izquierda,dice_arteria_derecha],run_eagerly=True)
    model.save("peso_test.h5")

    test_modelo("peso_test.h5",generador_test)
