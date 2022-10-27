from  keras import optimizers

def build_optimizer(optimizer,learning_rate):
    if optimizer == "sgd":
        optimizer = optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == "adam":
        optimizer = optimizers.Adam(learning_rate) 
    return optimizer
