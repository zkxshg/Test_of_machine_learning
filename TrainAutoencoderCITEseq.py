# https://gist.github.com/NikolayOskolkov/0b83ffcaf9b61c75f696babf539aa285#file-trainautoencoderciteseq-py
# Autoencoder training
estimator = autoencoder.fit([X_scRNAseq, X_scProteomics], 
                            [X_scRNAseq, X_scProteomics], 
                            epochs = 100, batch_size = 128, 
                            validation_split = 0.2, shuffle = True, verbose = 1)
print("Training Loss: ",estimator.history['loss'][-1])
print("Validation Loss: ",estimator.history['val_loss'][-1])
plt.plot(estimator.history['loss'])
plt.plot(estimator.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc = 'upper right')
plt.show()

# Encoder model
encoder = Model(input = [input_dim_scRNAseq, input_dim_scProteomics], 
                output = bottleneck)
bottleneck_representation = encoder.predict([X_scRNAseq, X_scProteomics])

# tSNE on Autoencoder bottleneck representation
model_tsne_auto = TSNE(learning_rate = 200, n_components = 2, random_state = 123, 
                       perplexity = 90, n_iter = 1000, verbose = 1)
tsne_auto = model_tsne_auto.fit_transform(bottleneck_representation)
plt.scatter(tsne_auto[:, 0], tsne_auto[:, 1], c = Y_scRNAseq, cmap = 'tab20', s = 10)
plt.title('tSNE on Autoencoder: Data Integration, CITEseq')
plt.xlabel("tSNE1")
plt.ylabel("tSNE2")
plt.show()
