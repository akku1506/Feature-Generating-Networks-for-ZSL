import tensorflow as tf
import numpy as np
import util
from sklearn.preprocessing import MinMaxScaler 
import sys
import os,os.path
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

def classificationLayer(x,classes,name="classification",reuse=False,isTrainable=True):
       
    with tf.variable_scope(name) as scope:
        
        if reuse:
            scope.reuse_variables()
        net = tf.layers.dense(inputs=x, units=classes,  \
                            kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02),  \
                            activation=None, name='fc1',trainable=isTrainable,reuse=reuse)

        net = tf.reshape(net, [-1, classes])    
    return net


class CLASSIFICATION2:
    # train_Y is interger 
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, logdir, modeldir, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=True):
        self.train_X =  _train_X 
        self.train_Y = _train_Y 
        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label 
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label 
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.shape[1]
        self.lr = _lr
        self.beta1 = _beta1
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.shape[0]
        self.logdir = logdir
        self.modeldir = modeldir
        ##########model_definition
        self.input = tf.placeholder(tf.float32,[self.batch_size, self.input_dim],name='input')
        self.label =  tf.placeholder(tf.int32,[self.batch_size],name='label')
        self.classificationLogits = classificationLayer(self.input,self.nclass)
        ############classification loss#########################

        self.classificationLoss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.classificationLogits, labels=self.label))
        classifierParams = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classification')

        for params in classifierParams:
            print (params.name)
        print ('...................')

        classifierOptimizer = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=self.beta1,beta2=0.999)
      
        classifierGradsVars = classifierOptimizer.compute_gradients(self.classificationLoss,var_list=classifierParams)    
        self.classifierTrain = classifierOptimizer.apply_gradients(classifierGradsVars)

        #################### what all to visualize  ############################
        tf.summary.scalar("ClassificationLoss",self.classificationLoss)
        for g,v in classifierGradsVars:    
            tf.summary.histogram(v.name,v)
            tf.summary.histogram(v.name+str('grad'),g)

        self.saver = tf.train.Saver()
        self.merged_all = tf.summary.merge_all()        

        if generalized:
            self.acc_seen, self.acc_unseen, self.H = self.fit()
            #print('Final: acc_seen=%.4f, acc_unseen=%.4f, h=%.4f' % (self.acc_seen, self.acc_unseen, self.H))
        else:
            self.acc = self.fit_zsl() 
    
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = np.random.permutation(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = np.random.permutation(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            if rest_num_examples > 0:
                return np.concatenate((X_rest_part, X_new_part), axis=0) , np.concatenate((Y_rest_part, Y_new_part), axis=0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]

    def fit(self):
        k=1
        best_H = 0
        best_seen = 0
        best_unseen = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(self.logdir, sess.graph)
            for epoch in range(self.nepoch):
                for i in range(0, self.ntrain, self.batch_size):
                    batch_input, batch_label = self.next_batch(self.batch_size)
                    _,loss,merged = sess.run([self.classifierTrain,self.classificationLoss,self.merged_all],feed_dict={self.input:batch_input,self.label:batch_label}) 
                    print ("Classification loss is:"+str(loss))
                
                    summary_writer.add_summary(merged, k)
                    k=k+1
                
                self.saver.save(sess, os.path.join(self.modeldir, 'models_'+str(epoch)+'.ckpt')) 
                print ("Model saved")
                acc_seen = 0
                acc_unseen = 0
                acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses,epoch)*100
                acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses,epoch)*100
                #print ('inside gzsl')
                #print (acc_seen,acc_unseen)
                H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
                #print('acc_seen=%.4f, acc_unseen=%.4f, h=%.4f' % (acc_seen, acc_unseen, H))
                if H > best_H:
                    best_seen = acc_seen
                    best_unseen = acc_unseen
                    best_H = H
        return best_seen, best_unseen, best_H

    def fit_zsl(self):
        k=1
        best_acc = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(self.logdir, sess.graph)
            for epoch in range(self.nepoch):
                for i in range(0, self.ntrain, self.batch_size):      
                    batch_input, batch_label = self.next_batch(self.batch_size)
                    _,loss,merged = sess.run([self.classifierTrain,self.classificationLoss,self.merged_all],feed_dict={self.input:batch_input,self.label:batch_label}) 
                    print ("Classification loss is:"+str(loss))
                
                    summary_writer.add_summary(merged, k)
                    k=k+1
                
                self.saver.save(sess, os.path.join(self.modeldir, 'models_'+str(epoch)+'.ckpt')) 
                print ("Model saved")
                acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses,epoch)*100
                #print (acc)
                if acc > best_acc:
                    best_acc = acc
        return best_acc 

    def val_gzsl(self, test_X, test_label, target_classes,epoch): 
        start = 0
        ntest = test_X.shape[0]
        predicted_label = np.empty_like(test_label)  

        self.input1 = tf.placeholder(tf.float32,[None, self.input_dim],name='test_features')
        self.classificationLogits = classificationLayer(self.input1,self.nclass,reuse=True,isTrainable=False)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='classification')
            
            self.saver = tf.train.Saver(var_list=params)
            
            for var in params:
                print (var.name+"\t")

            string = self.modeldir+'/models_'+str(epoch)+'.ckpt'
            #print (string) 
            #print (self.nepoch-1)
            try:
                self.saver.restore(sess, string)
            except:
                print("Previous weights not found of classifier") 
                sys.exit(0)

            print ("Model loaded")
            self.saver = tf.train.Saver()
            for i in range(0, ntest, self.batch_size):
                end = min(ntest, start+self.batch_size)
                output = sess.run([self.classificationLogits],feed_dict={self.input1:test_X[start:end]}) 
                #print (np.squeeze(np.array(output)).shape)
                predicted_label[start:end] = np.argmax(np.squeeze(np.array(output)), axis=1)
                start = end

            acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0.0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += np.sum(test_label[idx]==predicted_label[idx]) / np.sum(idx)
            #print ('yoo')
            #print (acc_per_class)
        acc_per_class /= target_classes.shape[0]
        return acc_per_class 

    def val(self, test_X, test_label, target_classes,epoch): 
        start = 0
        ntest = test_X.shape[0]
        predicted_label = np.empty_like(test_label)
        
        self.input1 = tf.placeholder(tf.float32,[None, self.input_dim],name='test_features')
 
        self.classificationLogits = classificationLayer(self.input1,self.nclass,reuse=True,isTrainable=False)
                
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='classification')
            
            self.saver = tf.train.Saver(var_list=params)
            
            for var in params:
                print (var.name+"\t")

            string = self.modeldir+'/models_'+str(epoch)+'.ckpt'
            #print (string) 
            #print (self.nepoch-1)
            try:
                self.saver.restore(sess, string)
            except:
                print("Previous weights not found of classifier") 
                sys.exit(0)

            print ("Model loaded")
            self.saver = tf.train.Saver()

            for i in range(0, ntest, self.batch_size):
                end = min(ntest, start+self.batch_size)
                output = sess.run([self.classificationLogits],feed_dict={self.input1:test_X[start:end]}) 
                #print (np.squeeze(np.array(output)).shape)
                predicted_label[start:end] = np.argmax(np.squeeze(np.array(output)), axis=1)
                start = end

            acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label, target_classes.shape[0])
        return acc

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = np.zeros(nclass)
        for i in range(0,nclass):
            idx = (test_label == i)
            acc_per_class[i] = np.sum(test_label[idx]==predicted_label[idx]) / np.sum(idx)
        return np.mean(acc_per_class) 