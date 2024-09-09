# ETPGSR  This work is developed based on the framework of TPGSR(https://github.com/mjq11302010044/TPGSR).
A DenseNet-RNN model is proposed as a text prior generator, along with improvements to the text prior feature transformer based on transposed convolution. A fusion module is introduced to combine text prior features with image super-resolution features, and a scene image degradation simulation method is proposed, which combines learnable neural operators and conventional operators to construct a high-resolution and low-resolution paired dataset. By pre-training on the simulated dataset and incorporating the improved text prior branch network structure, text recognition is performed on super-resolution reconstructed images using ASTER, MORAN, and CRNN. On the TextZoom dataset, accuracies of 64.5%, 60.8%, and 54.0% were achieved, respectively, surpassing several baseline models such as TPGSR and TATT. 


1. install python env as TPGSR. 
2. For test:  sh test.sh
3. For training: sh train_TPGSR-TSRN.sh, by modifying  the train_data_dir in ./config/super_resolution.yaml,  train the model with two steps:
    1) pre-train with synthetic HR-LR dataset (For making pre-training HR-LR dataset from Synth90k, run  ./dataset/create_lmdb2.py, the pretrained neural degradation operators are in ./pretrained_neural_degradation_operators/)
     2) train with TextZoom's training set 
