# Enhanced Text Prior Guided Text Image Super-Resolution(ETPGSR) 
   
1. install python env as TPGSR.
2. For training: sh train_TPGSR-TSRN.sh, by modifying  the train_data_dir in ./config/super_resolution.yaml,  train the model with two steps:
    1) pre-train with synthetic HR-LR dataset (For making pre-training HR-LR dataset from Synth90k, run  ./dataset/create_lmdb2.py, the pretrained neural degradation operators are in ./pretrained_neural_degradation_operators/)
     2) train with TextZoom's training set
3. For test:  sh test.sh 
The trained TP Generator, SR model and neural degradation operators can be downloaded from Baidu drive: https://pan.baidu.com/s/1V_E-faeC8LmDTu9k-fKANQ?pwd=tkf5
