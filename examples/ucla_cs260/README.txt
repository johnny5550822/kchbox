By King Chung Ho (Johnny)
Note: Each function are commented in details. If you have any questions, please contact me: johnny5550822@g.ucla.edu
1. The main functions are the core functions to run each classifier: main_ae, main_ae_knn,main_knn,main_nn. Within these main functions, you can control the hyperparameters for cross-validation.
2. There are four cross-validations functions (which can perform 10-fold or leave-one-out cross-validation). The name begins by “n_fold_cross_validation_xxx”, which xxx is the type of classifiers (ae=autoencoder, ae_knn = autoencoder+KNN, knn=KNN, nn=neural network)
3. “Kchbox_xxx” is the core function for each of the classifier. For NN and autoecnoder, they require . sub-functions from folders “neural_network” and “sparse_autoencoder”. NN and autoecnoder codes are vectorised (should be fast for medium size dataset).
4. “general_functions” folder contains useful functions to generate features, plot roc curve, etc 
5. “data” folder contains the data, which is in spreadsheets
6. Result is stored in 4 folders:  “10foldvalidation_result_dataset1”, “10foldvalidation_result_dataset2”, “leave-one-out-result_dataset1”, “leave-one-out-result_dataset2”
7. “result_temp” folder is for debugging (temporarily storing the result). 
