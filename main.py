#!/usr/bin/python
import os
import json
import logging
import time

#from keras.callbacks import ModelCheckpoint
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
from wandb.keras import WandbCallback
import wandb

from optimizadores import build_optimizer
from perdidas import dice_coefficient_loss
from metricas import dice_arteria_derecha, dice_arteria_izquierda, dice_arteria_principal, dice_coefficient, dice_promedio
from dataloader import DataLoader
from modelo import build_unet

from keras.models import load_model
from test_best import test_modelo
#wandb.init(settings=wandb.Settings(start_method="fork"))

import os

def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
#fix_gpu()




def main(config=None):
    tipo_de_dato = ".nrrd"
    cantidad_de_clases = 4
    dim=(128, 128, 64,1)

    #wandb.init(settings=wandb.Settings(start_method="fork"),
    #                    project="segmentacion3d_arteria_pulmonar_1_clase",
    #                    entity="aaronponce", config=config, tags=["unet3d","nrrd"],
    #                    notes="entrenamienro de unet con datos nrrd")

    #------------------------+
    # Creacion de artefactos 
    #------------------------+
    wandb.init(settings=wandb.Settings(start_method="fork"),config=config)
    print("wandb.config: ",wandb.config)

    config = wandb.config
    
    directorio_pesos = wandb.run.dir
    artefacto_dataloader = wandb.Artifact("dataloader", type="dataloader", description="preprocesado de mri dataset")
    artefacto_dataloader.add_file("./dataloader.py")

    artefacto_modelo = wandb.Artifact("modelo", type="modelo", description="arquitectura del modelo")
    artefacto_modelo.add_file("./modelo.py")

    artefacto_main = wandb.Artifact("entrenamiento", type="entrenamiento", description="entrenamiento de modelo")
    artefacto_main.add_file("./main.py")
    artefacto_main.add_dir(directorio_rutas, name="rutas dataset")

    artefacto_metrica = wandb.Artifact("metricas", type="metricas", description="calculo de metricas de entrenamiento y test")
    artefacto_metrica.add_file("./metricas.py")

    artefacto_optimizador = wandb.Artifact("optimizador", type="optimizador", description="optimizadores de entrenamiento")
    artefacto_optimizador.add_file("./optimizadores.py")

    artefacto_perdida = wandb.Artifact("perdidas", type="perdidas", description="funciones de perdida para entrenamiento")
    artefacto_perdida.add_file("./perdidas.py")

    
    wandb.run.log_artifact(artefacto_main)
    wandb.run.log_artifact(artefacto_modelo)
    wandb.run.log_artifact(artefacto_dataloader)
    wandb.run.log_artifact(artefacto_metrica)


    inicio = time.time()

    optim = build_optimizer(config["optimizer"],config["learning_rate"]) 
    model = build_unet(dim, n_classes=cantidad_de_clases)
    model.compile(optimizer = optim, loss=dice_coefficient_loss, metrics=[dice_promedio,dice_arteria_principal,dice_arteria_izquierda,dice_arteria_derecha],run_eagerly=True)

    print(model.summary())

    generador_entrenamiento = DataLoader(conjuntos["train"], imagenes, etiquetas,
                                        batch_size=config["batch_size"],
                                        dim=dim[0:3],
                                        shuffle=True,
                                        n_clases=cantidad_de_clases,
                                        canales=1,
                                        tipo_de_dato=tipo_de_dato)

    generador_test = DataLoader(conjuntos["test"], imagenes, etiquetas, 
                                        batch_size=len(conjuntos["test"]),
                                        dim=dim[0:3],
                                        shuffle=True,
                                        n_clases=cantidad_de_clases,
                                        canales=1,
                                        tipo_de_dato=tipo_de_dato)
    
    #ruta_save = directorio_pesos+'/version_'+str(config["version"])+"_epoca_{epoch:02d}_val_loss_{val_loss:.2f}_loss_{loss:.2f}_val_dice_{val_dice_coefficient:.2f}_train_dice_{dice_coefficient:.2f}.h5"
    #checkpoint = ModelCheckpoint(ruta_save, monitor="loss", verbose=1,save_best_only=True, mode="min", save_weights_only=False)
    history = model.fit(generador_entrenamiento,validation_data=generador_test,epochs=config["epochs"], 
                        workers=3, use_multiprocessing=False, verbose=1,callbacks=[WandbCallback(monitor="val_dice_promedio",mode="max",verbose=1)])#checkpoint ,
    
    file = open("history_seg_ap_01_.json", "w")
    json.dump(history.history, file)
    file.close()

    total = (time.time()-inicio) / 60
    print(total)
    logging.info("Fin: "+str(total))


    artefacto_best_modelo_entrenado = wandb.Artifact(
            "best-modelo-entrenado", type="best modelo",
            description="best Modelo unet3d entrenado",
            metadata=dict(config))
    

    print("wandb.run.dir: {}".format(wandb.run.dir))

    if os.path.exists(wandb.run.dir+"/model-best.h5"):

        test_modelo(wandb.run.dir+"/model-best.h5",conjuntos["test"], imagenes, etiquetas,dim)
        time.sleep(0.3)
        
        artefacto_best_modelo_entrenado.add_file("inputs.pkl")
        artefacto_best_modelo_entrenado.add_file("pred_etiqueta.pkl")
        artefacto_best_modelo_entrenado.add_file("true_etiqueta.pkl")
        artefacto_best_modelo_entrenado.add_file(wandb.run.dir+"/model-best.h5")

    artefacto_best_modelo_entrenado.add_file("history_seg_ap_01_.json")
    wandb.run.log_artifact(artefacto_best_modelo_entrenado)



if __name__ == "__main__":

    #configuracion de entrenamiento
    hacer_sweep = True
    config = {
        "modelo":"unet3d",
        "learning_rate": 0.1,
        "epochs": 15,
        "batch_size": 1,
        "clases":4,
        "version": 0,
        "name": "exp2-partes_de_arteria_pulmonar-local",
        "resumen:": "dataset arteria pulmonar",
        "extension": ".nrrd",
        "optimizer":"adam"
    }


    #configuracion de barrido
    sweep_config = {
        'method': 'random'
    }

    metric = {
        'name': 'val_dice_coefficient',
        'goal': 'maximize'   
    }

    parameters = {
        'optimizer': {
            'values': ['adam']
            },
        
        'epochs': {'values': [500]},
        'batch_size':{'values':[4]},

        'learning_rate':{
            'distribution': 'uniform',
            'max':0.1, 'min':0.01}
    }

    sweep_config['parameters'] = parameters
    sweep_config['metric'] = metric

    directorio_rutas = "../rutas_de_dataset/v2"

    conjuntos = directorio_rutas + "/ids.json"
    etiquetas = directorio_rutas + "/etiquetas.json"
    imagenes  = directorio_rutas + "/imagenes.json"


    logging.basicConfig(filename="train"+str(config["version"])+".log", level=logging.DEBUG)


    conjuntos = open(conjuntos)
    etiquetas = open(etiquetas)
    imagenes = open(imagenes)

    conjuntos = json.load(conjuntos)
    etiquetas = json.load(etiquetas)
    imagenes = json.load(imagenes)


    print(config)
    if hacer_sweep:
        sweep_id = wandb.sweep(sweep=sweep_config, project='my-first-sweep')
        wandb.agent(sweep_id, function=main, count=2)
    else:
        main(config)

