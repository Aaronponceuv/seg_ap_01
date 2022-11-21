from keras import backend as K
# Funcion de Perdidas y aciertos
import logging
import tensorflow as tf
import wandb
import numpy as np

#https://stackoverflow.com/questions/72195156/correct-implementation-of-dice-loss-in-tensorflow-keras

def dice_coefficient(y_true, y_pred):
    #tf.print("y_true.shape:",tf.shape(y_true))
    back_true = y_true[:,:,:,:,0]
    principal_true = y_true[:,:,:,:,1]
    derecha_true = y_true[:,:,:,:,2]
    izquierda_true = y_true[:,:,:,:,3]

    back_pred = y_pred[:,:,:,:,0]
    principal_pred = y_pred[:,:,:,:,1]
    derecha_pred = y_pred[:,:,:,:,2]
    izquierda_pred = y_pred[:,:,:,:,3]


    flat_back_true = K.flatten(back_true)
    flat_principal_true = K.flatten(principal_true) 
    flat_derecha_true = K.flatten(derecha_true)
    flat_izquierda_true = K.flatten(izquierda_true)

    flat_back_pred = K.flatten(back_pred)
    flat_principal_pred = K.flatten(principal_pred)
    flat_derecha_pred = K.flatten(derecha_pred)
    flat_izquierda_pred = K.flatten(izquierda_pred)

    smoothing_factor = 1
    dice_back = (2. * K.sum(flat_back_true * flat_back_pred)) / (K.sum(flat_back_true) + K.sum(flat_back_pred))
    dice_principal = (2. * K.sum(flat_principal_true * flat_principal_pred)) / (K.sum(flat_principal_true) + K.sum(flat_principal_pred))
    dice_derecha = (2. * K.sum(flat_derecha_true * flat_derecha_pred)) / (K.sum(flat_derecha_true) + K.sum(flat_derecha_pred))
    dice_izquierda = (2. * K.sum(flat_izquierda_true * flat_izquierda_pred)) / (K.sum(flat_izquierda_true) + K.sum(flat_izquierda_pred))


    wandb.log({"dice_background": dice_back.numpy()})
    wandb.log({"dice_principal": dice_principal.numpy()})
    wandb.log({"dice_derecha": dice_derecha.numpy()})
    wandb.log({"dice_izquierda": dice_izquierda.numpy()})

    dice_total = (dice_principal + dice_derecha + dice_izquierda)/3
    #tf.print("dice_total:",dice_total)
    
    wandb.log({"dice_total": dice_total.numpy()})
    return dice_total



def dice_arteria_principal(y_true, y_pred):
    arteria_principal_true = y_true[:,:,:,:,1]
    arteria_principal_pred = y_pred[:,:,:,:,1]

    flat_arteria_principal_true = K.flatten(arteria_principal_true)
    flat_arteria_principal_pred = K.flatten(arteria_principal_pred)

    interseccion = (2. * K.sum(flat_arteria_principal_true * flat_arteria_principal_pred))
    union = (K.sum(flat_arteria_principal_true) + K.sum(flat_arteria_principal_pred))


    if tf.equal(union, tf.constant([0.0]))[0].numpy():
        return tf.constant([0.0])[0]

    dice_arteria_principal = interseccion / union
    
    return dice_arteria_principal


def dice_arteria_izquierda(y_true, y_pred):
    print("dice_arteria_izquierda",flush=True)
    arteria_izquierda_true = y_true[:,:,:,:,3]
    arteria_izquierda_pred = y_pred[:,:,:,:,3]

    flat_arteria_izquierda_true = K.flatten(arteria_izquierda_true)
    flat_arteria_izquierda_pred = K.flatten(arteria_izquierda_pred)

    interseccion = (2. * K.sum(flat_arteria_izquierda_true * flat_arteria_izquierda_pred))
    union = (K.sum(flat_arteria_izquierda_true) + K.sum(flat_arteria_izquierda_pred))


    if tf.equal(union, tf.constant([0.0]))[0].numpy():
        return tf.constant([0.0])[0]

    dice_arteria_izquierda = interseccion / union

    return dice_arteria_izquierda


def dice_arteria_derecha(y_true, y_pred):
    arteria_derecha_true = y_true[:,:,:,:,2]
    arteria_derecha_pred = y_pred[:,:,:,:,2]

    flat_arteria_derecha_true = K.flatten(arteria_derecha_true)
    flat_arteria_derecha_pred = K.flatten(arteria_derecha_pred)

    interseccion = (2. * K.sum(flat_arteria_derecha_true * flat_arteria_derecha_pred))
    union = (K.sum(flat_arteria_derecha_true) + K.sum(flat_arteria_derecha_pred))


    if tf.equal(union, tf.constant([0.0]))[0].numpy():
        return tf.constant([0.0])[0]

    dice_arteria_derecha = interseccion / union

    
    return dice_arteria_derecha


def dice_background(y_true, y_pred):
    background_true = y_true[:,:,:,:,0]
    background_pred = y_pred[:,:,:,:,0]

    flat_background_true = K.flatten(background_true)
    flat_background_pred = K.flatten(background_pred)

    interseccion = (2. * K.sum(flat_background_true * flat_background_pred))
    union = (K.sum(flat_background_true) + K.sum(flat_background_pred))


    if tf.equal(union, tf.constant([0.0]))[0].numpy():
        return tf.constant([0.0])[0]

    dice_background = interseccion / union
    
    return dice_background


def dice_promedio(y_true, y_pred):
   dice_izquierda = dice_arteria_izquierda(y_true, y_pred)
   dice_derecha = dice_arteria_derecha(y_true, y_pred)
   dice_principal = dice_arteria_principal(y_true, y_pred)

   return (dice_izquierda+dice_derecha+dice_principal)/3



