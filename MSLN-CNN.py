# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 18:52:32 2017

@author: Administrator
"""
###############################################################################
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
import time
import scipy.io as sio
import os
from sklearn.decomposition import PCA
from pre import get_its,windowFeature,one_hot,contrary_one_hot,disorder,next_batch,conv2dlayer,aug_data,save_result,load_data
###############################################################################
data_name = 'Indian_pines'#'Indian_pines'#'PaviaU'#'Washington'#'Salinas'
data_norm, labels_ori, x_train, y_train, train_loc, x_test, y_test, test_loc=load_data(data_name)
dim_input = 3
dim_out = int(np.max(labels_ori))
dropout=tf.placeholder(dtype='float32')
batch_size = 128
w = 27
k=3
step = 1
epoch = 0
global_step=tf.Variable(step)
learn_rate=tf.train.exponential_decay(0.05, global_step,100,0.8, staircase=False)
display_step = 100
num_epoch = 500
y_train,train_loc,y_test,test_loc=aug_data(data_name,k,data_norm,labels_ori, dim_out, x_train, y_train, train_loc, x_test, y_test, test_loc)
###############################################################################
pca = PCA(n_components=dim_input)
data_PCA = pca.fit_transform(data_norm.reshape(data_norm.shape[0]*data_norm.shape[1], -1))
data_PCA = data_PCA.reshape(data_norm.shape[0], data_norm.shape[1],-1)

X_train = windowFeature(data_PCA, train_loc, w)
X_test = windowFeature(data_PCA, test_loc, w)

Y_train = one_hot(y_train,dim_out )
Y_test = one_hot(y_test,dim_out )

X_train,Y_train=disorder(X_train,Y_train)
X_test,Y_test=disorder(X_test,Y_test)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
###############################################################################
x = tf.placeholder(tf.float32,[None,w,w,dim_input],name='x_input')#placeholder 实例通常用来为算法的实际输入值作占位符，输入图片的大小
y = tf.placeholder(tf.float32,[None,dim_out],name='y_output')

weights={'W11':tf.Variable(tf.truncated_normal([1,1,dim_input,8],stddev=0.1)),
         
         'W12':tf.Variable(tf.truncated_normal([1,1,8,16],stddev=0.1)),
         
         'W13':tf.Variable(tf.truncated_normal([1,1,16,32],stddev=0.1)),
         
         'W14':tf.Variable(tf.truncated_normal([1,1,32,64],stddev=0.1)),

         'W15':tf.Variable(tf.truncated_normal([256,dim_out],stddev=0.1)),
         
         'W21':tf.Variable(tf.truncated_normal([3,3,dim_input,8],stddev=0.1)),
         
         'W22':tf.Variable(tf.truncated_normal([3,3,16,16],stddev=0.1)),
         
         'W23':tf.Variable(tf.truncated_normal([3,3,32,32],stddev=0.1)),
         
         'W24':tf.Variable(tf.truncated_normal([3,3,64,64],stddev=0.1)),
                                               
         'W27':tf.Variable(tf.truncated_normal([128,dim_out],stddev=0.1)),
         
         'W26':tf.Variable(tf.truncated_normal([144,128],stddev=0.1)),
         
         'W31':tf.Variable(tf.truncated_normal([5,5,dim_input,8],stddev=0.1)),
         
         'W32':tf.Variable(tf.truncated_normal([5,5,16,16],stddev=0.1)),
         
         'W33':tf.Variable(tf.truncated_normal([5,5,32,32],stddev=0.1)),
         
         'W34':tf.Variable(tf.truncated_normal([5,5,64,64],stddev=0.1)),
                                               
         'W37':tf.Variable(tf.truncated_normal([128,dim_out],stddev=0.1)),
         
         'W36':tf.Variable(tf.truncated_normal([144,128],stddev=0.1))
        }

bias={'B11':tf.Variable(tf.constant(0.1,shape=[8])),
      
      'B12':tf.Variable(tf.constant(0.1,shape=[16])),
      
      'B13':tf.Variable(tf.constant(0.1,shape=[32])),
      
      'B14':tf.Variable(tf.constant(0.1,shape=[64])),

      'B15':tf.Variable(tf.constant(0.1,shape=[dim_out])),
      
      'B21':tf.Variable(tf.constant(0.1,shape=[8])),
      
      'B22':tf.Variable(tf.constant(0.1,shape=[16])),
      
      'B23':tf.Variable(tf.constant(0.1,shape=[32])),
      
      'B24':tf.Variable(tf.constant(0.1,shape=[64])),
                                
      'B27':tf.Variable(tf.constant(0.1,shape=[dim_out])),
      
      'B26':tf.Variable(tf.constant(0.1,shape=[128])),
      
      'B31':tf.Variable(tf.constant(0.1,shape=[8])),
      
      'B32':tf.Variable(tf.constant(0.1,shape=[16])),
      
      'B33':tf.Variable(tf.constant(0.1,shape=[32])),
      
      'B34':tf.Variable(tf.constant(0.1,shape=[64])),
                                    
      'B37':tf.Variable(tf.constant(0.1,shape=[dim_out])),
                                               
      'B36':tf.Variable(tf.constant(0.1,shape=[128]))
      }
###############################################################################
x = tf.reshape(x,shape=[-1,w,w,dim_input])

conv11= conv2dlayer(x,weights['W11'],bias['B11'],[1,1,1,1])
pool11 = tf.nn.max_pool(conv11,[1,2,2,1],[1,2,2,1],padding='SAME', name='POOL1')
conv12 = conv2dlayer(pool11,weights['W12'],bias['B12'],[1,1,1,1])
pool12 = tf.nn.max_pool(conv12,[1,2,2,1],[1,2,2,1],padding='SAME', name='POOL1')
conv13 = conv2dlayer(pool12,weights['W13'],bias['B13'],[1,1,1,1])
pool13 = tf.nn.max_pool(conv13,[1,2,2,1],[1,2,2,1],padding='SAME', name='POOL1')
conv14 = conv2dlayer(pool13,weights['W14'],bias['B14'],[1,1,1,1])
pool14 = tf.nn.max_pool(conv14,[1,2,2,1],[1,2,2,1],padding='SAME', name='POOL1')
reshape1 = tf.reshape(pool14,[-1,pool14.get_shape().as_list()[1]*pool14.get_shape().as_list()[2]*pool14.get_shape().as_list()[3]])
f1 = tf.add(tf.matmul(reshape1,weights['W15']),bias['B15'])
y1=tf.nn.softmax(f1)

conv21= conv2dlayer(x,weights['W21'],bias['B21'],[1,1,1,1])
pool21 = tf.nn.max_pool(tf.concat( [conv21, conv11],3),[1,2,2,1],[1,2,2,1],padding='SAME', name='POOL1')
conv22 = conv2dlayer(pool21,weights['W22'],bias['B22'],[1,1,1,1])
pool22 = tf.nn.max_pool(tf.concat( [conv22, conv12],3),[1,2,2,1],[1,2,2,1],padding='SAME', name='POOL1')
conv23 = conv2dlayer(pool22,weights['W23'],bias['B23'],[1,1,1,1])
pool23 = tf.nn.max_pool(tf.concat( [conv23, conv13],3),[1,2,2,1],[1,2,2,1],padding='SAME', name='POOL1')
conv24 = conv2dlayer(pool23,weights['W24'],bias['B24'],[1,1,1,1])
pool24 = tf.nn.max_pool(tf.concat( [conv24, conv14],3),[1,2,2,1],[1,2,2,1],padding='SAME', name='POOL1')

pool210 = tf.nn.max_pool(tf.concat( [conv21, conv11],3),[1,32,32,1],[1,32,32,1],padding='SAME', name='POOL1')
pool220 = tf.nn.max_pool(tf.concat( [conv22, conv12],3),[1,16,16,1],[1,16,16,1],padding='SAME', name='POOL1')
pool230 = tf.nn.max_pool(tf.concat( [conv23, conv13],3),[1,8,8,1],[1,8,8,1],padding='SAME', name='POOL1')
pool240 = tf.nn.max_pool(tf.concat( [conv24, conv14],3),[1,4,4,1],[1,4,4,1],padding='SAME', name='POOL1')
pool2=tf.concat([pool210,pool240],3)
reshape2 = tf.reshape(pool2,[-1,pool2.get_shape().as_list()[1]*pool2.get_shape().as_list()[2]*pool2.get_shape().as_list()[3]])

f222=tf.nn.relu(tf.add(tf.matmul(reshape2,weights['W26']),bias['B26']))
f2 = tf.add(tf.matmul(f222,weights['W27']),bias['B27'])
y2=tf.nn.softmax(f2)

conv31= conv2dlayer(x,weights['W31'],bias['B31'],[1,1,1,1])
pool31 = tf.nn.max_pool(tf.concat( [conv31, conv11],3),[1,2,2,1],[1,2,2,1],padding='SAME', name='POOL1')
conv32 = conv2dlayer(pool31,weights['W32'],bias['B32'],[1,1,1,1])
pool32 = tf.nn.max_pool(tf.concat( [conv32, conv12],3),[1,2,2,1],[1,2,2,1],padding='SAME', name='POOL1')
conv33 = conv2dlayer(pool32,weights['W33'],bias['B33'],[1,1,1,1])
pool33 = tf.nn.max_pool(tf.concat( [conv33, conv13],3),[1,2,2,1],[1,2,2,1],padding='SAME', name='POOL1')
conv34 = conv2dlayer(pool33,weights['W34'],bias['B34'],[1,1,1,1])
pool34 = tf.nn.max_pool(tf.concat( [conv34, conv14],3),[1,2,2,1],[1,2,2,1],padding='SAME', name='POOL1')

pool310 = tf.nn.max_pool(tf.concat( [conv31, conv11],3),[1,32,32,1],[1,32,32,1],padding='SAME', name='POOL1')
pool320 = tf.nn.max_pool(tf.concat( [conv32, conv12],3),[1,16,16,1],[1,16,16,1],padding='SAME', name='POOL1')
pool330 = tf.nn.max_pool(tf.concat( [conv33, conv13],3),[1,8,8,1],[1,8,8,1],padding='SAME', name='POOL1')
pool340 = tf.nn.max_pool(tf.concat( [conv34, conv14],3),[1,4,4,1],[1,4,4,1],padding='SAME', name='POOL1')
pool3=tf.concat([pool310,pool340],3)
reshape3 = tf.reshape(pool3,[-1,pool3.get_shape().as_list()[1]*pool3.get_shape().as_list()[2]*pool3.get_shape().as_list()[3]])

f333=tf.nn.relu(tf.add(tf.matmul(reshape3,weights['W36']),bias['B36']))
f3 = tf.add(tf.matmul(f333,weights['W37']),bias['B37'])
y3=tf.nn.softmax(f3)

#y_=tf.multiply(0.5,tf.add(y2,y3))
y_=y2
###############################################################################
cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=f1, name=None))
cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=f2, name=None))
cross_entropy3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=f3, name=None))
cross_entropy=tf.add(cross_entropy2,cross_entropy3)
train_step = tf.train.RMSPropOptimizer(learn_rate).minimize(cross_entropy,global_step)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
###############################################################################
def get_oa(X_valid,Y_valid):
    size = np.shape(X_valid)
    num = size[0]
    index_all = 0
    step_ = 3000
    y_pred = []
    while index_all<num:
        if index_all + step_ > num:
            input = X_valid[index_all:, :, :, :]
        else:
            input = X_valid[index_all:(index_all+step_), :, :, :]
        index_all += step_
        temp1 = y_.eval(feed_dict={x: input,dropout:1.0})
        y_pred1=contrary_one_hot(temp1).astype('int32')
        y_pred.extend(y_pred1)
    y=contrary_one_hot(Y_valid).astype('int32')
    return y_pred,y
###############################################################################
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
index = batch_size
time_train_start=time.clock()
while epoch<num_epoch:
    batch_x,batch_y = next_batch(X_train,Y_train,index)
    sess.run(train_step, feed_dict={x: batch_x, y: batch_y,dropout: 0.5})
    index = index+batch_size
    if index>X_train.shape[0]:
        index = batch_size
        epoch=epoch+1
        X_train,Y_train=disorder(X_train,Y_train)   
    
    step += 1
    if step%display_step == 0:
        loss,acc = sess.run([cross_entropy,accuracy], feed_dict={x: batch_x, y: batch_y,dropout: 0.5})
        print('step %d,training accuracy %f,cross_entrop %f'%(step,acc,loss))       
        y_pr,y_tr = get_oa(X_test,Y_test)
        oa = accuracy_score(y_tr,y_pr)
        print('valid accuracy %f'%(oa))

time_train_end=time.clock()   
print("Optimization Finished!")
###############################################################################
time_test_start=time.clock()#计算测试程序开始
if data_name == 'Indian_pines':
    y_pred=y_.eval(feed_dict={x:X_test,y:Y_test,dropout: 1})
    y_pred=contrary_one_hot(y_pred).astype('int32')
elif data_name == 'PaviaU':
    y_pred=np.array([])
    for i in range(20):
        temp=y_.eval(feed_dict={x: X_test[i*1800:(i+1)*1800], y: Y_test[i*1800:(i+1)*1800],dropout:1})
        pred=contrary_one_hot(temp)
        y_pred=np.hstack((y_pred,pred))
    y_pr=y_.eval(feed_dict={x: X_test[36000:], y: Y_test[36000:], dropout: 1})
    y_pr=contrary_one_hot(y_pr).astype('int32')
    y_pred=np.hstack((y_pred,y_pr))
elif data_name == 'Washington':
    y_pred=np.array([])
    for i in range(10):
        temp=y_.eval(feed_dict={x: X_test[i*1000:(i+1)*1000], y: Y_test[i*1000:(i+1)*1000],dropout:1})
        pred=contrary_one_hot(temp)
        y_pred=np.hstack((y_pred,pred))
    y_pr=y_.eval(feed_dict={x: X_test[10000:], y: Y_test[10000:], dropout: 1})
    y_pr=contrary_one_hot(y_pr).astype('int32')
    y_pred=np.hstack((y_pred,y_pr))
elif data_name == 'Salinas':
    y_pred=np.array([])
    for i in range(20):
        temp=y_.eval(feed_dict={x: X_test[i*2000:(i+1)*2000], y: Y_test[i*2000:(i+1)*2000],dropout:1})
        pred=contrary_one_hot(temp)
        y_pred=np.hstack((y_pred,pred))
    y_pr=y_.eval(feed_dict={x: X_test[40000:], y: Y_test[40000:], dropout: 1})
    y_pr=contrary_one_hot(y_pr).astype('int32')
    y_pred=np.hstack((y_pred,y_pr))
    
y=contrary_one_hot(Y_test).astype('int32')
oa = accuracy_score(y,y_pred)
per_class_acc=recall_score(y,y_pred,average=None)
aa=np.mean(per_class_acc)
kappa=cohen_kappa_score(y,y_pred)
time_test_end=time.clock()
###############################################################################
print('oa',oa)
print('aa',aa)
print('kappa',kappa)
save_result(data_name,oa,aa,kappa,per_class_acc,time_train_end-time_train_start,time_test_end-time_test_start)

t=0
for i in range(np.shape(labels_ori)[0]):
    for j in range(np.shape(labels_ori)[1]):
        if t==0:
            label_index=np.array([[i],[j]])
            t=1
        elif t==1:
            label_index=np.concatenate((label_index,np.array([[i],[j]])),axis=1) 

X_data=windowFeature(data_PCA, label_index, w)
temp = y_.eval(feed_dict={x: X_data})
y_pred=contrary_one_hot(temp).astype('int32')
print(y_pred)
    
plot_max=np.zeros(np.shape(labels_ori))
index=0
for i in range(np.shape(labels_ori)[0]):
    for j in range(np.shape(labels_ori)[1]):
        plot_max[i][j]=y_pred[index]
        index+=1
path=os.getcwd()
sio.savemat(path+'/plot/'+data_name+'_'+'plot', {'plot_max':plot_max})
    
sess.close()
