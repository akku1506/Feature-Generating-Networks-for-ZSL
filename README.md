# Feature-Generating-Networks-for-ZSL
This repository is an implementation of Feature Generating Networks for Zero Shot Learning (https://arxiv.org/pdf/1712.00981.pdf) in Tensorflow.

## Pre-Requisites
1. Compatible with Python-3, Tensorflow 1.5
2. Download datasets for zero shot learning from http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip

## Running the code

### Learning the Softmax Classifier
```
python classifier.py --manualSeed 9182 --preprocessing --lr 0.001 --image_embedding res101 --class_embedding att --nepoch 50 --dataset AWA1 --batch_size 100 --attSize 85 --resSize 2048 --modeldir models_classifier --logdir logs_classifier --dataroot '/home/test/notebooks/transductive+learning/xlsa17/data'
```
### Running the discriminator generator and learning final classifier on Zero Shot Learning
```
python clswgan.py --manualSeed 9182 --cls_weight 0.01 --preprocessing --val_every 1 --lr 0.00001 --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 30 --syn_num 200 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset AWA1 --batch_size 64 --nz 85 --attSize 85 --resSize 2048 --modeldir models_awa --logdir logs_awa --dataroot '/home/test/notebooks/transductive+learning/xlsa17/data' --classifier_modeldir './models_classifier'  --classifier_checkpoint 49 
```
### Running the discriminator generator and learning final classifier on Generalised Zero Shot Learning
```
python clswgan.py --manualSeed 9182 --cls_weight 0.01 --preprocessing --val_every 1 --lr 0.00001 --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 30 --syn_num 2400 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset AWA1 --batch_size 64 --nz 85 --attSize 85 --resSize 2048 --modeldir models_awa --logdir logs_awa --dataroot '/home/test/notebooks/transductive+learning/xlsa17/data' --classifier_modeldir './models_classifier'  --classifier_checkpoint 49 --gzsl 
```
