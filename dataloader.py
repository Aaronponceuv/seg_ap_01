from scipy.interpolate import RegularGridInterpolator
from matplotlib.font_manager import json_load
from scipy import io
import numpy as np
import os
from scipy import io
import keras
import json
import logging
import h5py
import nrrd

class DataLoader(keras.utils.Sequence):
    def __init__(self, lista_ID, imagenes,etiquetas, batch_size=10, dim=(140,160,60,1), shuffle=True, n_clases=1,canales=1,tipo_de_dato=".mat", trans_3dpcmra=None):

        try:
            assert len(lista_ID) >= batch_size, "error!: batch size es mayor a la cantidad de datos disponibles"
        except AssertionError as e:
            msj = "dataloader line 19 {}".format(e)
            logging.error(msj)

        self.imagenes = imagenes
        self.etiquetas = etiquetas
        self.list_IDs = lista_ID
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.clases = n_clases
        self.cantidad_canales = canales
        self.trans_3dpcmra = trans_3dpcmra
        self.tipo_de_dato = tipo_de_dato
        self.on_epoch_end()


    def on_epoch_end(self):
        #Genera indices para el batch
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indices)


    def __len__(self):
        # retorna la cantidad de batches por epoca o interacione
        try:
            return int(np.floor(len(self.list_IDs) / self.batch_size))
        except Exception as e:
            logging.debug("__len__: {} {}".format(e, e.args))


    def __getitem__(self,index): # index = numero de batch hasta __len__
        #print("index: {}".format(index))

        #indices seleccionados
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        #Busqueda de lista de ids
        list_IDs_temp = [self.list_IDs[k] for k in indices]

        X, y = self.__data_generation(list_IDs_temp)
        
        return X,y

    
    def __data_generation(self,lista_ids_temp):
        X = np.empty((self.batch_size,*self.dim,self.cantidad_canales))
        y = np.empty((self.batch_size,*self.dim,self.clases))

        #generacion de datos
        for i, ID in enumerate(lista_ids_temp):
            ruta_imagen = self.imagenes[ID] 
            ruta_etiqueta = self.etiquetas[ID]
            logging.info("{} {} {}".format(ID, ruta_imagen, ruta_etiqueta))

            try:
                if(self.tipo_de_dato == ".mat"):
                    data = io.loadmat(ruta_imagen)

                    if(self.trans_3dpcmra):
                        magnitud =  data["MR_FFE_AP"]
                        magnitud = self.pcmra3d(data)
                    else:
                        magnitud =  data["data"]

                    magnitud = self.rediminesionar(magnitud)
                    magnitud = np.expand_dims(magnitud, axis=3) # simil a [...,None]

                    data = io.loadmat(ruta_etiqueta)
                    etiqueta = data["SEG"]
                    #etiqueta = etiqueta[1:-1,1:-1,1:-1]

                if(self.tipo_de_dato == ".nrrd"):
                    magnitud, _ = nrrd.read(ruta_imagen)
                    magnitud = self.rediminesionar(magnitud)
                    magnitud = np.expand_dims(magnitud, axis=3)
                    etiqueta, _ = nrrd.read(ruta_etiqueta)
                    #etiqueta = self.rediminesionar(etiqueta)
                    #etiqueta = np.expand_dims(etiqueta, axis=3)
            except Exception as e:
                print("ERROR L101 Dataloader: {} {} {} {}".format(ID,ruta_etiqueta,e,e.args))
                logging.critical("ERROR L101 Dataloader: {} {} {} {}".format(ID,ruta_etiqueta,e,e.args))
            
            etiqueta = keras.utils.to_categorical(etiqueta, num_classes=self.clases)
            etiqueta = self.rediminesionar(etiqueta)
            etiqueta = (etiqueta > 0).astype(int)

            X[i,] = magnitud
            y[i,] = etiqueta

        return X,y

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


    def pcmra3d(self,data):
        magnitud =  data["data"]["MR_FFE_AP"]
        velocidad_fh = data["data"]["MR_PCA_FH"] # x
        velocidad_ap = data["data"]["MR_PCA_AP"] # y
        velocidad_rl = data["data"]["MR_PCA_RL"]

        N = magnitud.shape[3]

        producto = 0
        for t in range(N):
            producto = producto + magnitud[:,:,:,t]*(velocidad_ap[:,:,:,t]**2+velocidad_fh[:,:,:,t]**2+velocidad_rl[:,:,:,t]**2)    

        pcmra3d = np.sqrt(producto/N)
        return pcmra3d



#"""
if __name__ == "__main__":
    logging.basicConfig(filename="train.log", level=logging.DEBUG)

    #conjuntos = "dic_dataset/ids.json"
    #etiquetas= "dic_dataset/etiquetas.json"
    #imagenes = "dic_dataset/imagenes.json"

    conjuntos = "dataset_nrrd/ids.json"
    etiquetas= "dataset_nrrd/etiquetas.json"
    imagenes = "dataset_nrrd/imagenes.json"

    conjuntos = "../dataset_nrrd/ids.json"
    etiquetas = "../dataset_nrrd/etiquetas.json"
    imagenes  = "../dataset_nrrd/imagenes.json"

    conjunto_entrenamiento = open(conjuntos)
    etiquetas = open(etiquetas)
    imagenes = open(imagenes)

    conjunto_entrenamiento = json.load(conjunto_entrenamiento)
    etiquetas = json.load(etiquetas)
    imagenes = json.load(imagenes)
    dat = DataLoader(conjunto_entrenamiento["test"],"test", imagenes, etiquetas,batch_size=5, dim=(128, 128, 60) , shuffle=True, n_clases=4,canales=1,tipo_de_dato=".nrrd")

    import matplotlib.pyplot as plt
    for i in range(10):
        for X,y in dat:
            print("y.shape:",y.shape)
            """
            X = X[0].reshape(128, 128, 60)
            y = y[0].reshape(128, 128, 60,4)

            fig, ax = plt.subplots(3,3, figsize=(20,20))
            coronal = 50
            axial = 45
            ax[0,0].imshow(X[:,:,30],cmap="gray")
            ax[0,1].imshow(X[:,coronal,:],cmap="gray")
            ax[0,2].imshow(X[axial,:,:],cmap="gray")

            ax[0,0].plot([0,127],[axial,axial], color="white", linewidth=2)
            ax[0,0].plot([coronal,coronal],[0,127], color="green", linewidth=2)

            ax[1,0].imshow(X[:,:,30],cmap="gray")
            ax[1,1].imshow(X[:,coronal,:],cmap="gray")
            ax[1,2].imshow(X[axial,:,:],cmap="gray")

            ax[1,0].imshow(y[:,:,30,1],cmap="jet",alpha=0.18)
            ax[1,1].imshow(y[:,coronal,:,1],cmap="jet",alpha=0.18)
            ax[1,2].imshow(y[axial,:,:,1],cmap="jet",alpha=0.18)

            ax[1,0].plot([0,127],[axial,axial], color="white", linewidth=2)
            ax[1,0].plot([coronal,coronal],[0,127], color="green", linewidth=2)

            ax[2,0].imshow(y[:,:,30,1])
            ax[2,1].imshow(y[:,coronal,:,1])
            ax[2,2].imshow(y[axial,:,:,1])

            ax[2,0].plot([0,127],[axial,axial], color="white", linewidth=2)
            ax[2,0].plot([coronal,coronal],[0,127], color="green", linewidth=2)
            plt.show()
            """

        #"""