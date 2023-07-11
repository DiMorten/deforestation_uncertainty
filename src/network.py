from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model, Sequential
from src.evidential_learning import DirichletLayer

def resnet_block(x, n_filter, ind):
    x_init = x

    ## Conv 1
    x = Conv2D(n_filter, (3, 3), activation='relu', padding="same", name = 'res1_net'+str(ind))(x)
    x = Dropout(0.5, name = 'drop_net'+str(ind))(x)
    ## Conv 2
    x = Conv2D(n_filter, (3, 3), activation='relu', padding="same", name = 'res2_net'+str(ind))(x)
    
    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), activation='relu', padding="same", name = 'res3_net'+str(ind))(x_init)
    
    ## Add
    x = Add()([x, s])
    return x

def resnet_block_dropout(x, n_filter, ind):
    x_init = x

    ## Conv 1
    x = Conv2D(n_filter, (3, 3), activation='relu', padding="same", name = 'res1_net'+str(ind))(x)
    x = Dropout(0.5, name = 'drop_net'+str(ind))(x, training=True)
    ## Conv 2
    x = Conv2D(n_filter, (3, 3), activation='relu', padding="same", name = 'res2_net'+str(ind))(x)
    
    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), activation='relu', padding="same", name = 'res3_net'+str(ind))(x_init)
    
    ## Add
    x = Add()([x, s])
    return x

# Residual U-Net model
def build_resunet(input_shape, nb_filters, n_classes, last_activation='softmax'):
    '''Base network to be shared (eq. to feature extraction)'''
    input_layer= Input(shape = input_shape, name="input_enc_net")
    
    res_block1 = resnet_block(input_layer, nb_filters[0], 1)
    pool1 = MaxPool2D((2 , 2), name='pool_net1')(res_block1)
    
    res_block2 = resnet_block(pool1, nb_filters[1], 2) 
    pool2 = MaxPool2D((2 , 2), name='pool_net2')(res_block2)
    
    res_block3 = resnet_block(pool2, nb_filters[2], 3) 
    pool3 = MaxPool2D((2 , 2), name='pool_net3')(res_block3)
    
    res_block4 = resnet_block(pool3, nb_filters[2], 4)
    
    res_block5 = resnet_block(res_block4, nb_filters[2], 5)
    
    res_block6 = resnet_block(res_block5, nb_filters[2], 6)
    
    upsample3 = Conv2D(nb_filters[2], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net3')(UpSampling2D(size = (2,2))(res_block6))
    
    merged3 = concatenate([res_block3, upsample3], name='concatenate3')

    upsample2 = Conv2D(nb_filters[1], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net2')(UpSampling2D(size = (2,2))(merged3))
                                                 
    merged2 = concatenate([res_block2, upsample2], name='concatenate2')
                                                                                          
    upsample1 = Conv2D(nb_filters[0], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net1')(UpSampling2D(size = (2,2))(merged2))
    merged1 = concatenate([res_block1, upsample1], name='concatenate1')

    output = Conv2D(n_classes,(1,1), activation = last_activation, padding = 'same', name = 'output')(merged1)
                                                                                                           
    return Model(input_layer, output)

def build_resunet_dropout(input_shape, nb_filters, n_classes):
    '''Base network to be shared (eq. to feature extraction)'''
    input_layer= Input(shape = input_shape, name="input_enc_net")
    
    res_block1 = resnet_block_dropout(input_layer, nb_filters[0], 1)
    pool1 = MaxPool2D((2 , 2), name='pool_net1')(res_block1)
    
    res_block2 = resnet_block_dropout(pool1, nb_filters[1], 2) 
    pool2 = MaxPool2D((2 , 2), name='pool_net2')(res_block2)
    
    res_block3 = resnet_block_dropout(pool2, nb_filters[2], 3) 
    pool3 = MaxPool2D((2 , 2), name='pool_net3')(res_block3)
    
    res_block4 = resnet_block_dropout(pool3, nb_filters[2], 4)
    
    res_block5 = resnet_block_dropout(res_block4, nb_filters[2], 5)
    
    res_block6 = resnet_block_dropout(res_block5, nb_filters[2], 6)
    
    upsample3 = Conv2D(nb_filters[2], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net3')(UpSampling2D(size = (2,2))(res_block6))
    
    upsample3 = Dropout(0.5)(upsample3, training=True)

    merged3 = concatenate([res_block3, upsample3], name='concatenate3')

    upsample2 = Conv2D(nb_filters[1], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net2')(UpSampling2D(size = (2,2))(merged3))

    upsample2 = Dropout(0.5)(upsample2, training=True)

    merged2 = concatenate([res_block2, upsample2], name='concatenate2')
                                                                                          
    upsample1 = Conv2D(nb_filters[0], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net1')(UpSampling2D(size = (2,2))(merged2))
    upsample1 = Dropout(0.5)(upsample1, training=True)

    merged1 = concatenate([res_block1, upsample1], name='concatenate1')

    output = Conv2D(n_classes,(1,1), activation = 'softmax', padding = 'same', name = 'output')(merged1)
                                                                                                           
    return Model(input_layer, output)



def resnet_block_spatial_dropout(x, n_filter, dropout_seed, ind, training=True):
    x_init = x

    ## Conv 1
    x = Conv2D(n_filter, (3, 3), activation='relu', padding="same", name = 'res1_net'+str(ind))(x)
    # x = Dropout(0.5, name = 'drop_net'+str(ind))(x, training = True)
    x = SpatialDropout2D(0.25, name = 'drop_net'+str(ind), seed = dropout_seed)(x, training = training)

    ## Conv 2
    x = Conv2D(n_filter, (3, 3), activation='relu', padding="same", name = 'res2_net'+str(ind))(x)
    
    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), activation='relu', padding="same", name = 'res3_net'+str(ind))(x_init)
    
    ## Add
    x = Add()([x, s])
    return x

# Residual U-Net model
def build_resunet_dropout_spatial(input_shape, nb_filters, n_classes, dropout_seed = None, last_activation='softmax', training=True):
    '''Base network to be shared (eq. to feature extraction)'''

    dropout = 0.25
    
    input_layer= Input(shape = input_shape, name="input_enc_net")
    
    res_block1 = resnet_block_spatial_dropout(input_layer, nb_filters[0], dropout_seed, 1, training=training)
    pool1 = MaxPool2D((2 , 2), name='pool_net1')(res_block1)
    
    res_block2 = resnet_block_spatial_dropout(pool1, nb_filters[1], dropout_seed, 2, training=training) 
    pool2 = MaxPool2D((2 , 2), name='pool_net2')(res_block2)
    
    res_block3 = resnet_block_spatial_dropout(pool2, nb_filters[2], dropout_seed, 3, training=training) 
    pool3 = MaxPool2D((2 , 2), name='pool_net3')(res_block3)
    
    res_block4 = resnet_block_spatial_dropout(pool3, nb_filters[2], dropout_seed, 4, training=training)
    
    res_block5 = resnet_block_spatial_dropout(res_block4, nb_filters[2], dropout_seed, 5, training=training)
    
    res_block6 = resnet_block_spatial_dropout(res_block5, nb_filters[2], dropout_seed, 6, training=training)
    
    upsample3 = Conv2D(nb_filters[2], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net3')(UpSampling2D(size = (2,2))(res_block6))
    
    upsample3 = SpatialDropout2D(dropout, seed = dropout_seed)(upsample3, training=training)

    merged3 = concatenate([res_block3, upsample3], name='concatenate3')

    upsample2 = Conv2D(nb_filters[1], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net2')(UpSampling2D(size = (2,2))(merged3))

    upsample2 = SpatialDropout2D(dropout, seed = dropout_seed)(upsample2, training=training)

    merged2 = concatenate([res_block2, upsample2], name='concatenate2')
                                                                                          
    upsample1 = Conv2D(nb_filters[0], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net1')(UpSampling2D(size = (2,2))(merged2))

    upsample1 = SpatialDropout2D(dropout, seed = dropout_seed)(upsample1, training=training)

    merged1 = concatenate([res_block1, upsample1], name='concatenate1')

    output = Conv2D(n_classes,(1,1), activation = last_activation, padding = 'same', name = 'output')(merged1)
                                                                                                           
    return Model(input_layer, output)

# Evidential residual U-Net model
def build_evidential_resunet(input_shape, nb_filters, n_classes, dropout_seed = None, 
                             last_activation='softmax', training=True):
    '''Base network to be shared (eq. to feature extraction)'''

    dropout = 0.25
    
    input_layer= Input(shape = input_shape, name="input_enc_net")
    
    res_block1 = resnet_block_spatial_dropout(input_layer, nb_filters[0], dropout_seed, 1, training=training)
    pool1 = MaxPool2D((2 , 2), name='pool_net1')(res_block1)
    
    res_block2 = resnet_block_spatial_dropout(pool1, nb_filters[1], dropout_seed, 2, training=training) 
    pool2 = MaxPool2D((2 , 2), name='pool_net2')(res_block2)
    
    res_block3 = resnet_block_spatial_dropout(pool2, nb_filters[2], dropout_seed, 3, training=training) 
    pool3 = MaxPool2D((2 , 2), name='pool_net3')(res_block3)
    
    res_block4 = resnet_block_spatial_dropout(pool3, nb_filters[2], dropout_seed, 4, training=training)
    
    res_block5 = resnet_block_spatial_dropout(res_block4, nb_filters[2], dropout_seed, 5, training=training)
    
    res_block6 = resnet_block_spatial_dropout(res_block5, nb_filters[2], dropout_seed, 6, training=training)
    
    upsample3 = Conv2D(nb_filters[2], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net3')(UpSampling2D(size = (2,2))(res_block6))
    
    upsample3 = SpatialDropout2D(dropout, seed = dropout_seed)(upsample3, training=training)

    merged3 = concatenate([res_block3, upsample3], name='concatenate3')

    upsample2 = Conv2D(nb_filters[1], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net2')(UpSampling2D(size = (2,2))(merged3))

    upsample2 = SpatialDropout2D(dropout, seed = dropout_seed)(upsample2, training=training)

    merged2 = concatenate([res_block2, upsample2], name='concatenate2')
                                                                                          
    upsample1 = Conv2D(nb_filters[0], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net1')(UpSampling2D(size = (2,2))(merged2))

    upsample1 = SpatialDropout2D(dropout, seed = dropout_seed)(upsample1, training=training)

    merged1 = concatenate([res_block1, upsample1], name='concatenate1')

    output = Conv2D(n_classes,(1,1), activation = last_activation, padding = 'same', name = 'output')(merged1)

    output = DirichletLayer(n_classes)(output)                                                                                             

    return Model(input_layer, output)
