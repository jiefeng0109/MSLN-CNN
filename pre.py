# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 21:40:54 2018

@author: EGGSHELL
"""

import scipy.io as sio
import os
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import numpy as np
import tensorflow as tf

def local_constraint(x_test,train_loc,test_loc,y_train,dim_out,k):
    dist=np.zeros([x_test.shape[0],len(train_loc[0])])
    ppp=np.zeros([x_test.shape[0],])
    for t in range(x_test.shape[0]):
        dist[t,:]=np.sqrt((np.tile(test_loc[0][t], (1, len(train_loc[0])))-train_loc[0][:])**(2)+(np.tile(test_loc[1][t], (1, len(train_loc[0])))-train_loc[1][:])**(2))
    
    for i in range(x_test.shape[0]):
        new_loc=sorted(range(len(dist[i,:])), key=dist[i,:].__getitem__)
        new_lab=[y_train[i] for i in new_loc[0:k]]
        
        num_lab=np.zeros([dim_out,1])
        for j in range(len(new_lab)):
            num_lab[new_lab[j]-1]=num_lab[new_lab[j]-1]+1
        [pp,qq]=np.where(num_lab==max(num_lab))
        
        if new_lab[0]==1 or new_lab[0]==7 or new_lab[0]==9:
            ppp[i]=new_lab[0]
        if max(num_lab)==k:
            ppp[i]=pp+1

    return ppp
    
def nonlocal_constraint(x_test,train_loc,y_train,x_train,dim_out,k):
    spec=np.zeros([x_test.shape[0],len(train_loc[0])])
    qqq=np.zeros([x_test.shape[0],])
    for t in range(x_test.shape[0]):
        spec[t,:]=(np.sqrt((np.tile(x_test[t][:], (len(train_loc[0]),1))-x_train)**(2))).sum(axis=1)

    for i in range(x_test.shape[0]): 
        new_loc=sorted(range(len(spec[i,:])), key=spec[i,:].__getitem__)        
        new_lab=[y_train[i] for i in new_loc[0:k]]
        
        num_lab=np.zeros([dim_out,1])
        for j in range(len(new_lab)):
            num_lab[new_lab[j]-1]=num_lab[new_lab[j]-1]+1
        [pp,qq]=np.where(num_lab==max(num_lab))
    
        if new_lab[0]==1 or new_lab[0]==7 or new_lab[0]==9:
            qqq[i]=new_lab[0]
        if max(num_lab)==k:
            qqq[i]=pp+1

    return qqq
    
def aug_data(data_name,k,data_norm,labels_ori, dim_out, x_train, y_train, train_loc, x_test, y_test, test_loc):
    
    ppp=local_constraint(x_test,train_loc,test_loc,y_train,dim_out,k)
    qqq=nonlocal_constraint(x_test,train_loc,y_train,x_train,dim_out,k)
    
    x_loc=[]
    y_loc=[]
    new_train_loc=[]
    new_y_train=[]
    index=[]
    x_test_loc=[]
    y_text_loc=[]
    new_test_loc=[]
    new_y_test=[]

    plot_max=np.zeros(np.shape(labels_ori))

    for i in range(x_test.shape[0]):
        if qqq[i]==ppp[i] and qqq[i]!=0:
            a=np.shape(np.where(np.array(new_y_train)==qqq[i])[0])[0]
            b=np.shape(np.where(np.array(y_train)==qqq[i])[0])[0]

            if a<b+1:
                x_loc.append(test_loc[0,i])
                y_loc.append(test_loc[1,i])
                new_y_train=np.hstack((new_y_train,qqq[i]))
                plot_max[test_loc[0,i],test_loc[1,i]]=qqq[i]
                index.append(i)
    
    new_train_loc=np.vstack((x_loc,y_loc))
    new_y_train=new_y_train.astype('int32')   
    train_loc=np.hstack([train_loc,new_train_loc])   
    y_train=np.hstack((y_train,new_y_train))   
    
    t=0  
    for i in range(x_test.shape[0]):
        if i==index[t]:
            t=t+1
            if index[t] == index[-1]:
                t=0
            continue
        x_test_loc.append(test_loc[0,i])
        y_text_loc.append(test_loc[1,i])
        new_y_test=np.hstack((new_y_test,y_test[i]))
    
    new_test_loc=np.vstack((x_test_loc,y_text_loc))
    new_y_test=new_y_test.astype('int32') 
    
    print(accuracy_score(labels_ori[tuple(new_train_loc)],new_y_train))
    print(np.shape(x_loc))
#    print(np.shape(x_test_loc))
#    print(new_y_train)
 
#    path=os.getcwd()
#    sio.savemat(path+'/plot/'+data_name+'_'+'plot', {'plot_max':plot_max})
#    sio.savemat(path+'/data/'+data_name+'/'+data_name+'_pre_aug', {'train_x':x_train,
#                'train_y':y_train, 'train_loc':train_loc, 'test_x':x_test,
#                'test_y':new_y_test, 'test_loc':new_test_loc, 'data_norm':data_norm,
#                'labels_ori':labels_ori})
    

    return y_train,train_loc,new_y_test,new_test_loc
    
def readData(data_name):
    ''' 读取原始数据和标准类标 '''
    path = os.getcwd()+'/data/'+data_name
    if data_name == 'Indian_pines':
        data = sio.loadmat(path+'/Indian_pines_corrected.mat')['indian_pines_corrected']
        labels = sio.loadmat(path+'/Indian_pines_gt.mat')['indian_pines_gt']
    elif data_name == 'PaviaU':
        data = sio.loadmat(path+'/PaviaU.mat')['paviaU']
        labels = sio.loadmat(path+'/PaviaU_gt.mat')['paviaU_gt']
    elif data_name == 'Washington':
        data = sio.loadmat(path+'/washington_datax.mat')['washington_datax']
        labels = sio.loadmat(path+'/washington_labelx.mat')['washington_labelx']
        labels=labels-1
    elif data_name == 'KSC':
        data = sio.loadmat(path+'/KSC.mat')['KSC']
        labels = sio.loadmat(path+'/KSC_gt.mat')['KSC_gt']
    elif data_name == 'Salinas':
        data = sio.loadmat(path+'/Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat(path+'/Salinas_gt.mat')['salinas_gt']
    data = np.float64(data)
    labels = np.array(labels).astype(float)

    return data,labels

def normalizeData(data):
    ''' 原始数据归一化处理（每条） '''
    data_norm = np.zeros(np.shape(data))
    for i in range(np.shape(data)[0]):
        for j in range(np.shape(data)[1]):
            data_norm[i,j,:] = preprocessing.normalize(data[i,j,:].reshape(1,-1))[0]
    return data_norm
    
def selectTrainTest(data, labels, p):
    ''' 从所有类中每类选取训练样本和测试样本 '''
    c = int(labels.max())
    x = np.array([], dtype=float).reshape(-1, data.shape[2])  # 训练样本
    xb = []
    x_loc1 = []
    x_loc2 = []
    x_loc = []
    y = np.array([], dtype=float).reshape(-1, data.shape[2])#(0,200)
    yb = []
    y_loc1 = []
    y_loc2 = []
    y_loc = []
    for i in range(1, c+1):
        loc1, loc2 = np.where(labels == i)
        num = len(loc1)
        order = np.random.permutation(range(num))
        loc1 = loc1[order]
        loc2 = loc2[order]
        num1 = int(np.round(num*p))
        x = np.vstack([x, data[loc1[:num1], loc2[:num1], :]])#[:2]前2个数,x=(2,200)
        y = np.vstack([y, data[loc1[num1:], loc2[num1:], :]])#[2:]从0,1,2，……数，2到最后，y=(44,200)
        xb.extend([i]*num1)#[1,1]
        yb.extend([i]*(num-num1))#[1,1……，1]，46个1
        x_loc1.extend(loc1[:num1])
        x_loc2.extend(loc2[:num1])
        y_loc1.extend(loc1[num1:])
        y_loc2.extend(loc2[num1:])
        x_loc = np.vstack([x_loc1, x_loc2])#（2,2）
        y_loc = np.vstack([y_loc1, y_loc2])#（2,44）
    return x, xb, x_loc, y, yb, y_loc  #x=(512,200),xb=512个标签,x_loc=(2,512),y=(9737,200),yb=9737个标签，y_loc=(2,9737)

def get_its(data_name):
    '''得到最终的分割数据结果'''
    data_ori,labels_ori = readData(data_name)
    data_norm = normalizeData(data_ori)
    if data_name == 'Indian_pines':
        p = 0.05
    elif data_name == 'PaviaU':
        p = 0.03
    if data_name == 'Washington':
        p = 0.05
    elif data_name == 'KSC':
        p = 0.05
    elif data_name == 'Salinas':
        p = 0.01
    x_train, y_train, train_loc, x_test, y_test, test_loc = selectTrainTest(data_norm, labels_ori, p)
    print(int(np.max(labels_ori)))
    print(np.shape(x_train))

    path = os.getcwd()
    sio.savemat(path+'/data/'+data_name+'/'+data_name+'_pre', {'train_x':x_train,
                'train_y':y_train, 'train_loc':train_loc, 'test_x':x_test,
                'test_y':y_test, 'test_loc':test_loc, 'data_norm':data_norm,
                'labels_ori':labels_ori})
    
    return data_norm, labels_ori, x_train, y_train, train_loc, x_test, y_test, test_loc
    
def load_data(data_name):
    '''读取数据'''
    path = os.getcwd()
    pre = sio.loadmat(path + '/data/' + data_name + '/' + data_name + '_pre.mat')
    
    data_norm = pre['data_norm']
    labels_ori = pre['labels_ori']
    x_train = pre['train_x']
    y_train = pre['train_y'][0]
    train_loc = pre['train_loc']
    x_test = pre['test_x']
    y_test = pre['test_y'][0]
    test_loc = pre['test_loc']
    
    return data_norm,labels_ori,x_train,y_train,train_loc,x_test,y_test,test_loc
    
def windowFeature(data, loc, w ):
    '''从扩展矩阵中得到窗口特征'''
    size = np.shape(data)
    print(size)
    data_expand = np.zeros((int(size[0]+w-1),int(size[1]+w-1),size[2]))
    newdata = np.zeros((len(loc[0]), w, w,size[2]))
    for j in range(size[2]):    
        data_expand[:,:,j] = np.lib.pad(data[:,:,j], ((int(w / 2), int(w / 2)), (int(w / 2),int(w / 2))), 'symmetric')
        newdata[:,:,:,j] = np.zeros((len(loc[0]), w, w))
        for i in range(len(loc[0])):
            loc1 = loc[0][i]
            loc2 = loc[1][i]
            f = data_expand[loc1:loc1 + w, loc2:loc2 + w,j]
#            print(loc1,loc2)
            newdata[i, :, :,j] = f
    return newdata

def one_hot(lable,class_number):
    '''标签变为独热编码形式'''
    one_hot_array = np.zeros([len(lable),class_number])
    for i in range(len(lable)):  #range（5）等价于range（0， 5）是[0, 1, 2, 3, 4]没有5，range（0， 5） 等价于 range(0, 5, 1)每次跳跃的间距默认为1
        one_hot_array[i,int(lable[i]-1)] = 1
    return one_hot_array

def contrary_one_hot(label):
    '''反独热编码'''
    size=len(label)#len(label)表示行的长度，len(label[0])表示列的数量
    label_ori=np.empty(size)
    for i in range(size):
        label_ori[i]=np.argmax(label[i])+1  #argmax返回的是最大数的索引
    return label_ori

def disorder(X_origin,Y_origin):
    '''打乱数组顺序'''
    index_train = np.arange(X_origin.shape[0])  #（512，）,0~511，arange(0,1,0.1)，array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9]) 
    np.random.shuffle(index_train)  #随机打乱函数
    X_now = X_origin[index_train, :]
    Y_now = Y_origin[index_train, :]
    return X_now,Y_now

def next_batch(image,lable,batch_size):
    '''数据分批'''
    start = batch_size-100
    end = batch_size
    return image[start:end,:,:,:],lable[start:end]

def batch_norm(x):
    '''BN层'''
    return tf.contrib.layers.batch_norm(x,decay=0.9,epsilon=1e-5,scale=True,is_training=True)    

def conv2dlayer(x,W,B,stride):
    '''CONV层'''
    x = tf.nn.conv2d(x,W,stride,padding='SAME',name='CONV')#x是图片的所有参数，W是此卷积层的权重padding参数用来控制图片的边距，’SAME’表示卷积后的图片与原图片大小相同
    h = tf.nn.bias_add(x,B)
#    bn = batch_norm(h, is_training=True,is_conv_out=True,decay = 0.999)
    bn = batch_norm(h)
    convout = tf.nn.relu(bn)# 使用ReLu激活函数激活
    return convout

def conv3dlayer(x,W,B,stride):
    '''3D-CONV层'''
    x = tf.nn.conv3d(x,W,stride,padding='SAME',name='3D-CONV')
    h = tf.nn.bias_add(x,B)
    convout =tf.nn.relu(h)
    return convout
    
def save_result(data_name,oa,aa,kappa,per_class_acc,train_time,test_time):
    '''将实验结果保存在txt文件中'''
    write_content='\n'+data_name+'\n'+'oa:'+str(oa)+' aa:'+str(aa)+' kappa:'+str(kappa)+'\n'+'per_class_acc:'+str(per_class_acc)+'\n'+'train_time:'+str(train_time)+' test_time:'+str(test_time)+'\n'
    f = open(os.getcwd()+'/实验结果.txt','a')
    f.writelines(write_content)
    f.close()
    return








