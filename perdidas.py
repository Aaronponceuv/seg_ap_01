from keras import backend as K


def dice_arteria_principal(y_true, y_pred):
    arteria_principal_true = y_true[:,:,:,:,1]
    arteria_principal_pred = y_pred[:,:,:,:,1]

    flat_arteria_principal_true = K.flatten(arteria_principal_true)
    flat_arteria_principal_pred = K.flatten(arteria_principal_pred)

    dice_arteria_principal = (2. * K.sum(flat_arteria_principal_true * flat_arteria_principal_pred)) / (K.sum(flat_arteria_principal_true) + K.sum(flat_arteria_principal_pred))
    
    return dice_arteria_principal


def dice_arteria_izquierda(y_true, y_pred):
    arteria_izquierda_true = y_true[:,:,:,:,3]
    arteria_izquierda_pred = y_pred[:,:,:,:,3]

    flat_arteria_izquierda_true = K.flatten(arteria_izquierda_true)
    flat_arteria_izquierda_pred = K.flatten(arteria_izquierda_pred)

    dice_arteria_izquierda = (2. * K.sum(flat_arteria_izquierda_true * flat_arteria_izquierda_pred)) / (K.sum(flat_arteria_izquierda_true) + K.sum(flat_arteria_izquierda_pred))
    
    return dice_arteria_izquierda


def dice_arteria_derecha(y_true, y_pred):
    arteria_derecha_true = y_true[:,:,:,:,2]
    arteria_derecha_pred = y_pred[:,:,:,:,2]

    flat_arteria_derecha_true = K.flatten(arteria_derecha_true)
    flat_arteria_derecha_pred = K.flatten(arteria_derecha_pred)

    interseccion = K.sum(flat_arteria_derecha_true * flat_arteria_derecha_pred)
    dice_arteria_derecha = (2. * interseccion) / (K.sum(flat_arteria_derecha_true) + K.sum(flat_arteria_derecha_pred))
    
    return dice_arteria_derecha


def dice_background(y_true, y_pred):
    background_true = y_true[:,:,:,:,0]
    background_pred = y_pred[:,:,:,:,0]

    flat_background_true = K.flatten(background_true)
    flat_background_pred = K.flatten(background_pred)

    dice_background = (2. * K.sum(flat_background_true * flat_background_pred)) / (K.sum(flat_background_true) + K.sum(flat_background_pred))
    
    return dice_background


def dice_promedio(y_true, y_pred):
   dice_izquierda = dice_arteria_izquierda(y_true, y_pred)
   dice_derecha = dice_arteria_derecha(y_true, y_pred)
   dice_principal = dice_arteria_principal(y_true, y_pred)

   return (dice_izquierda+dice_derecha+dice_principal)/3

def dice_coefficient_loss(y_true, y_pred):    
    loss = 1 - dice_promedio(y_true, y_pred)
    return loss
