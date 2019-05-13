# https://gist.github.com/NikolayOskolkov/933e1db8ea9c2573a02dd8257ddcfe63
import numpy as np
import pandas as pd
from umap import UMAP
from keras.models import Model
import matplotlib.pyplot as plt
from keras.layers.merge import concatenate
from keras.layers import Input, Dense, Dropout

################## READ AND TRANSFORM DATA ##################
scRNAseq = pd.read_csv('scRNAseq.txt',sep='\t')
scBSseq = pd.read_csv('scBSseq.txt',sep='\t')
scATACseq = pd.read_csv('scATACseq.txt',sep='\t')

X_scRNAseq = scRNAseq.values[:,0:(scRNAseq.shape[1]-1)]
Y_scRNAseq = scRNAseq.values[:,scRNAseq.shape[1]-1]
X_scBSseq = scBSseq.values[:,0:(scBSseq.shape[1]-1)]
Y_scBSseq = scBSseq.values[:,scBSseq.shape[1]-1]
X_scATACseq = scATACseq.values[:,0:(scATACseq.shape[1]-1)]
Y_scATACseq = scATACseq.values[:,scATACseq.shape[1]-1]

X_scRNAseq = np.log(X_scRNAseq + 1)
X_scBSseq = np.log(X_scBSseq + 1)
X_scATACseq = np.log(X_scATACseq + 1)

######################## AUTOENCODER ########################
# Input Layer
ncol_scRNAseq = X_scRNAseq.shape[1]
input_dim_scRNAseq = Input(shape = (ncol_scRNAseq, ), name = "scRNAseq")
ncol_scBSseq = X_scBSseq.shape[1]
input_dim_scBSseq = Input(shape = (ncol_scBSseq, ), name = "scBSseq")
ncol_scATACseq = X_scATACseq.shape[1]
input_dim_scATACseq = Input(shape = (ncol_scATACseq, ), name = "scATACseq")

encoding_dim_scRNAseq = 30
encoding_dim_scBSseq = 30
encoding_dim_scATACseq = 30

# Dropout on Input Layer
dropout_scRNAseq = Dropout(0.2, name = "Dropout_scRNAseq")(input_dim_scRNAseq)
dropout_scBSseq = Dropout(0.2, name = "Dropout_scBSseq")(input_dim_scBSseq)
dropout_scATACseq = Dropout(0.2, name = "Dropout_scATACseq")(input_dim_scATACseq) 

# Encoder layer for each OMIC
encoded_scRNAseq = Dense(encoding_dim_scRNAseq, activation = 'elu', 
                         name = "Encoder_scRNAseq")(dropout_scRNAseq)
encoded_scBSseq = Dense(encoding_dim_scBSseq, activation = 'elu', 
                        name = "Encoder_scBSseq")(dropout_scBSseq)
encoded_scATACseq = Dense(encoding_dim_scATACseq, activation = 'elu', 
                          name = "Encoder_scATACseq")(dropout_scATACseq)

# Merging Encoder layers from different OMICs
merge = concatenate([encoded_scRNAseq, encoded_scBSseq, encoded_scATACseq])

# Bottleneck compression
bottleneck = Dense(50, kernel_initializer = 'uniform', activation = 'linear', 
                   name = "Bottleneck")(merge)

#Inverse merging
merge_inverse = Dense(encoding_dim_scRNAseq + encoding_dim_scBSseq + 
                      encoding_dim_scATACseq, 
                      activation = 'elu', name = "Concatenate_Inverse")(bottleneck)

# Decoder layer for each OMIC
decoded_scRNAseq = Dense(ncol_scRNAseq, activation = 'sigmoid', 
                         name = "Decoder_scRNAseq")(merge_inverse)
decoded_scBSseq = Dense(ncol_scBSseq, activation = 'sigmoid', 
                        name = "Decoder_scBSseq")(merge_inverse)
decoded_scATACseq = Dense(ncol_scATACseq, activation = 'sigmoid', 
                          name = "Decoder_scATACseq")(merge_inverse)

# Combining Encoder and Decoder into an Autoencoder model
autoencoder = Model(input = [input_dim_scRNAseq, input_dim_scBSseq, 
                             input_dim_scATACseq], 
                    output = [decoded_scRNAseq, decoded_scBSseq, decoded_scATACseq])

# Compile Autoencoder
autoencoder.compile(optimizer = 'adam', 
                    loss={'Decoder_scRNAseq': 'mean_squared_error', 
                          'Decoder_scBSseq': 'binary_crossentropy', 
                          'Decoder_scATACseq': 'binary_crossentropy'})
autoencoder.summary()

# Autoencoder training
estimator = autoencoder.fit([X_scRNAseq, X_scBSseq, X_scATACseq], 
                            [X_scRNAseq, X_scBSseq, X_scATACseq], epochs = 130, 
                            batch_size = 16, validation_split = 0.2, 
                            shuffle = True, verbose = 1)
print("Training Loss: ",estimator.history['loss'][-1])
print("Validation Loss: ",estimator.history['val_loss'][-1])
plt.plot(estimator.history['loss']); plt.plot(estimator.history['val_loss'])
plt.title('Model Loss'); plt.ylabel('Loss'); plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc = 'upper right')

# Encoder model
encoder = Model(input = [input_dim_scRNAseq, input_dim_scBSseq, 
                         input_dim_scATACseq], output = bottleneck)
bottleneck_representation = encoder.predict([X_scRNAseq, X_scBSseq, X_scATACseq])

############### UNIFORM MANIFOLD APPROXIMATION AND PROJECTION (UMAP) ###############
model_umap = UMAP(n_neighbors = 11, min_dist = 0.1, n_components = 2)
umap = model_umap.fit_transform(bottleneck_representation)
plt.scatter(umap[:, 0], umap[:, 1], c = Y_scRNAseq, cmap = 'tab10', s = 10)
plt.title('UMAP on Autoencoder: Data Integration, scNMTseq')
plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
