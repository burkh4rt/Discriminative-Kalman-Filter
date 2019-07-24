## Preprocessing

We preprocess the Flint dataset in the following way:

1. Download data [here](https://portal.nersc.gov/project/crcns/download/dream/data_sets/Flint_2012).
2. Run [flint_gather_data.m](flint_gather_data.m) to move the data from the separate files and nested structs into a single struct.
3. Run [flint_preprocess_data.m](flint_preprocess_data.m) to coarsen data and hit the neural data with PCA and z-scoring.  After the preprocessing, many sets were only ~6000 pairs long; so each run used 5000 training points and 1000 test points.
4. Run [flint_filtering.m](flint_filtering.m) to do the actual filtering.  An additional preprocessing step within the filtering code uses [radial_transform.m](radial_transform.m) to learn a radial transform with the training data and apply it to the testing data.
