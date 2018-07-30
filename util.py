from __future__ import print_function
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import sys

def map_label(label, classes):
    mapped_label =  np.empty_like(label)
    for i in range(classes.shape[0]):
        mapped_label[label==classes[i]] = i    

    return mapped_label


class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            self.read_matdataset(opt)
            #if opt.dataset == 'imageNet1K':
            #    self.read_matimagenet(opt)
            #else:
            #    self.read_matdataset(opt)

        self.index_in_epoch = 0
        self.epochs_completed = 0
                


    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
    
        self.attribute = matcontent['att'].T.astype(np.float32) 
        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                
                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = _train_feature.astype(np.float32)
                mx = self.train_feature.max()
                self.train_feature*(1/mx)
                self.train_label = label[trainval_loc].astype(np.int) 
                self.test_unseen_feature = _test_unseen_feature.astype(np.float32)
                self.test_unseen_feature*(1/mx)
                self.test_unseen_label = label[test_unseen_loc].astype(np.int) 
                self.test_seen_feature = _test_seen_feature.astype(np.float32) 
                self.test_seen_feature*(1/mx)
                self.test_seen_label = label[test_seen_loc].astype(np.int)

                print(self.train_feature.shape)
                print(self.train_label.shape)
                print(self.test_seen_feature.shape)
                print(self.test_unseen_feature.shape)
                print(self.test_seen_label.shape)
                print(self.test_unseen_label.shape)
                print(".....................")
            else:
                self.train_feature = feature[trainval_loc].astype(np.float32)
                self.train_label = label[trainval_loc].astype(np.int) 
                self.test_unseen_feature = feature[test_unseen_loc].astype(np.float32)
                self.test_unseen_label = label[test_unseen_loc].astype(np.int) 
                self.test_seen_feature = feature[test_seen_loc].astype(np.float32) 
                self.test_seen_label = label[test_seen_loc].astype(np.int)

        else:
            self.train_feature = feature[train_loc].astype(np.float32)
            self.train_label = label[train_loc].astype(np.int)
            self.test_unseen_feature = feature[val_unseen_loc].astype(np.float32)
            self.test_unseen_label = label[val_unseen_loc].astype(np.int) 
    
        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_unseen_label)
        self.ntrain = (self.train_feature).shape[0]
        self.ntrain_class = (self.seenclasses).shape[0]
        self.ntest_class = (self.unseenclasses).shape[0]
        self.train_class = np.copy(self.seenclasses)
        self.allclasses = np.arange(0, self.ntrain_class+self.ntest_class).astype(np.int)
        
        self.train_mapped_label = map_label(self.train_label, self.seenclasses) 

    
    def next_batch(self, batch_size):
        idx = np.random.permutation(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att
    """

    def read_matimagenet(self, opt):
        if opt.preprocessing:
            print('MinMaxScaler...')
            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = scaler.fit_transform(np.array(matcontent['features']))
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = scaler.transform(np.array(matcontent['features_val']))
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()
            matcontent = h5py.File('/BS/xian/work/data/imageNet21K/extract_res/res101_1crop_2hops_t.mat', 'r')
            feature_unseen = scaler.transform(np.array(matcontent['features']))
            label_unseen = np.array(matcontent['labels']).astype(int).squeeze() - 1
            matcontent.close()
        else:
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = np.array(matcontent['features'])
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = np.array(matcontent['features_val'])
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()


        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
        self.attribute = torch.from_numpy(matcontent['w2v']).np.float32()
        self.train_feature = torch.from_numpy(feature).np.float32()
        self.train_label = torch.from_numpy(label).np.int() 
        self.test_seen_feature = torch.from_numpy(feature_val).np.float32()
        self.test_seen_label = torch.from_numpy(label_val).np.int() 
        self.test_unseen_feature = torch.from_numpy(feature_unseen).np.float32()
        self.test_unseen_label = torch.from_numpy(label_unseen).np.int() 
        self.ntrain = self.train_feature.size()[0]
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0 
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]] 


    # select batch samples by randomly drawing batch_size classes    
    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.np.intTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]
            
        batch_feature = torch.np.float32Tensor(batch_size, self.train_feature.size(1))       
        batch_label = torch.np.intTensor(batch_size)
        batch_att = torch.np.float32Tensor(batch_size, self.attribute.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]] 
        return batch_feature, batch_label, batch_att
    """