using  HDF5

elmo_embeddings_path= "data/elmo/elmo-layers.mrp-amr-training.hdf5"
elmo_layer_no = 1

embeddings = h5open(elmo_embeddings_path, "r") do file
    read(file)
end

