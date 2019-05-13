# https://gist.github.com/NikolayOskolkov/277d65621267658e71d06eb59b577e44#file-autoencoderciteseq-py
# Input Layer

ncol_scRNAseq = X_scRNAseq.shape[1]

input_dim_scRNAseq = Input(shape = (ncol_scRNAseq, ), name = "scRNAseq")

ncol_scProteomics = X_scProteomics.shape[1]

input_dim_scProteomics = Input(shape = (ncol_scProteomics, ), name = "scProteomics")



# Dimensions of Encoder for each OMIC

encoding_dim_scRNAseq = 50

encoding_dim_scProteomics = 10



# Encoder layer for each OMIC

encoded_scRNAseq = Dense(encoding_dim_scRNAseq, activation = 'linear', 

                         name = "Encoder_scRNAseq")(input_dim_scRNAseq)

encoded_scProteomics = Dense(encoding_dim_scProteomics, activation = 'linear', 

                             name = "Encoder_scProteomics")(input_dim_scProteomics)



# Merging Encoder layers from different OMICs

merge = concatenate([encoded_scRNAseq, encoded_scProteomics])



# Bottleneck compression

bottleneck = Dense(50, kernel_initializer = 'uniform', activation = 'linear', 

                   name = "Bottleneck")(merge)



#Inverse merging

merge_inverse = Dense(encoding_dim_scRNAseq + encoding_dim_scProteomics, 

                      activation = 'elu', name = "Concatenate_Inverse")(bottleneck)



# Decoder layer for each OMIC

decoded_scRNAseq = Dense(ncol_scRNAseq, activation = 'sigmoid', 

                         name = "Decoder_scRNAseq")(merge_inverse)

decoded_scProteomics = Dense(ncol_scProteomics, activation = 'sigmoid', 

                             name = "Decoder_scProteomics")(merge_inverse)



# Combining Encoder and Decoder into an Autoencoder model

autoencoder = Model(input = [input_dim_scRNAseq, input_dim_scProteomics], 

                    output = [decoded_scRNAseq, decoded_scProteomics])



# Compile Autoencoder

autoencoder.compile(optimizer = 'adam', 

                    loss={'Decoder_scRNAseq': 'mean_squared_error', 

                          'Decoder_scProteomics': 'mean_squared_error'})

autoencoder.summary()
