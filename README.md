# k-nearest-neighbors
```k-nearest-neighbors``` implements kNN to classify text documents. 

Args: 
* ```training_data```: text file with vectors to train on
* ```test_data```: text file with vectors to test on 
* ```k_val```: the number of nearest neighbors
* ```similarity_func```: similarity function ID number. If 1, uses Euclidean distance. If 2, uses cosine similarity. 

Returns: 
* ```sys_output```: the classification result on the training and test data (cf. sys_output under examples)
* ```acc_file```: the confusion matrix and acccuracy for the training and test data

To run: 
```
src/build_kNN.sh input/train.vectors.txt input/test.vectors.txt k_val similarity_func output/sys_output > output/acc_file
```

HW4 OF LING572 (01/31/2023)