## 2019.0127 说明：使用SBP成像后的图像作为初始图像，真实分布作为标签，利用CNN进行训练，
## 训练结果在训练集（cross)上可以得到0.5的误差。##
## ！！！！ 2019.0514 说明：最后运行部分务必参考 ert_cnngan_shixian_onlypat2tri0415_lossimportant

# 函数库导入
from __future__ import print_function
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import numpy as np 
import csv 
import argparse
import time


# 初始化
tf.reset_default_graph()
#from tensorflow.python.framework import ops
#ops.reset_default_graph()
parser = argparse.ArgumentParser(description='')   # 解析命令行参数和选项的标准模块
EPS = 1e-12 # EPS用于保证log函数里面的参数大于零
EPS1 = 1e-3
EPS12 = 1e-3
EPS2 = 1e-3
num_pixels = 2209
TI = 47
time_start=time.time()


# 从csv文件中读取数据，读取为矩阵格式
## 电导率分布
Zl_pat2tri_imshow = np.loadtxt(open("./DeepData_2020/zl_imshow_nbubl.csv","rb"),delimiter=",",skiprows=0)
## SBP成像值
zl_sbp_pat2tri = np.loadtxt(open("./DeepData_2020/Cytzl_imshow_New.csv","rb"),delimiter=",",skiprows=0)
zl_sbp_shiyan = np.loadtxt(open("./DeepData_2020/Cytzl_imshow_New_shiyan.csv","rb"),delimiter=",",skiprows=0)
zl_sbp_coms = np.loadtxt(open("./DeepData_2020/Cytzl_imshow_New_coms1.csv","rb"),delimiter=",",skiprows=0)

Zl_pat2tritrain_imshow = Zl_pat2tri_imshow[1:14000]   # [1:1000] 
zl_sbp_pat2tritrain = zl_sbp_pat2tri[1:14000] 
Zl_pat2tritest_imshow = Zl_pat2tri_imshow[14501:21500]  # [1:1000] 
zl_sbp_pat2tritest = zl_sbp_pat2tri[14501:21500] 

# 程序模块
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]  Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,3,3,1], padding='SAME')

def batch_norm(xx):
    fc_mean, fc_var = tf.nn.moments(xx, axes=[0],)
    scale = tf.Variable(tf.ones([1]))
    shift = tf.Variable(tf.zeros([1]))
    epsilon = 0.001
    xx = tf.nn.batch_normalization(xx, fc_mean, fc_var, shift, scale, epsilon)
    return xx

def generator(image, keep_prob, reuse, name="generator"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        ## conv1 layer ##
        W_conv1 = weight_variable([3,3, 1,8]) # patch 3x3, in size 1, out size 8
        b_conv1 = bias_variable([8])
        h_conv1 = conv2d(image, W_conv1) + b_conv1 # output size 47x47x8
        h_pool1 = max_pool_2x2(h_conv1)                        # output size 24x24x8        
        h_lere1 = tf.nn.leaky_relu(h_pool1)
#        h_batn1 = batch_norm(h_lere1)
        ## conv2 layer ##
        W_conv2 = weight_variable([3,3, 8, 32]) # patch 3x3, in size 8, out size 32
        b_conv2 = bias_variable([32])
        h_conv2 = conv2d(h_lere1, W_conv2) + b_conv2 # output size 24x24x32
        h_pool2 = max_pool_2x2(h_conv2)                          # output size 12x12x32        
        h_lere2 = tf.nn.leaky_relu(h_pool2)
#        h_batn2 = batch_norm(h_lere2)       
        ## conv3 layer ##
        W_conv3 = weight_variable([3,3, 32, 128]) # patch 3x3, in size 8, out size 128
        b_conv3 = bias_variable([128])
        h_conv3 = conv2d(h_lere2, W_conv3) + b_conv3 # output size 12x12x128
        h_pool3 = max_pool_3x3(h_conv3)                          # output size 4x4x64        
        h_lere3 = tf.nn.leaky_relu(h_pool3)
#        h_batn3 = batch_norm(h_lere3)
        ## fc1 layer ##
        W_fc1 = weight_variable([4*4*128, 2048])
        b_fc1 = bias_variable([2048])
        # [n_samples, 3, 3, 64] ->> [n_samples, 3*3*64]
        h_pool2_flat = tf.reshape(h_lere3, [-1, 4*4*128])
        h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)        
        h_fc1_lere = tf.nn.leaky_relu(h_fc1_drop)
        h_fc1_batn = batch_norm(h_fc1_lere)
        ## fc2 layer ##
        W_fc2 = weight_variable([2048, 2209])
        b_fc2 = bias_variable([2209])
        h_fc2 = tf.matmul(h_fc1_batn, W_fc2) + b_fc2
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)        
        h_fc2_lere = tf.nn.leaky_relu(h_fc2_drop) 
        h_fc2_batn = batch_norm(h_fc2_lere)
        ## fc3 layer ##
        W_fc3 = weight_variable([2209, 2209])
        b_fc3 = bias_variable([2209])
        prediction_gen = tf.nn.tanh(tf.matmul(h_fc2_batn, W_fc3) + b_fc3)
        return (prediction_gen)

def discriminator(image, targets, keep_prob, reuse=False, name="discriminator"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        image_con = tf.concat([image, targets], 3)
        ## conv1 layer ##
        W_conv1 = weight_variable([5,5,2,8]) # patch 5x5, in size 1, out size 32
        b_conv1 = bias_variable([8])
        h_conv1 = conv2d(image_con, W_conv1) + b_conv1 # output size 28x28x32
        h_pool1 = max_pool_2x2(h_conv1)                      # output size 14x14x32 
        h_lere1 = tf.nn.leaky_relu(h_pool1)
#        h_batn1 = batch_norm(h_lere1)
        ## conv2 layer ##
        W_conv2 = weight_variable([5,5, 8, 32]) # patch 5x5, in size 32, out size 64
        b_conv2 = bias_variable([32])
        h_conv2 = tf.nn.relu(conv2d(h_lere1, W_conv2) + b_conv2) # output size 14x14x64
        h_pool2 = max_pool_2x2(h_conv2)                         # output size 7x7x64
        h_lere2 = tf.nn.leaky_relu(h_pool2)
#        h_batn2 = batch_norm(h_lere2)
        ## conv3 layer ##
        W_conv3 = weight_variable([5,5, 32, 128]) # patch 5x5, in size 8, out size 64
        b_conv3 = bias_variable([128])
        h_conv3 = conv2d(h_lere2, W_conv3) + b_conv3 # output size 9x9x128
        h_pool3 = max_pool_3x3(h_conv3)                          # output size 3x3x128        
        h_lere3 = tf.nn.leaky_relu(h_pool3)
#        h_batn3 = batch_norm(h_lere3)
        ## fc1 layer ##
        W_fc1 = weight_variable([4*4*128, 2048])  # 9*9*32  3*3*128
        b_fc1 = bias_variable([2048])
        # [n_samples, 3, 3, 64] ->> [n_samples, 3*3*64]
        h_pool2_flat = tf.reshape(h_lere3, [-1, 4*4*128])
        h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)        
        h_fc1_lere = tf.nn.leaky_relu(h_fc1_drop)
        h_fc1_batn = batch_norm(h_fc1_lere)
        ## fc2 layer ##
        W_fc2 = weight_variable([2048, 1024])
        b_fc2 = bias_variable([1024])
        h_fc2 = tf.matmul(h_fc1_batn, W_fc2) + b_fc2
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)        
        h_fc2_lere = tf.nn.leaky_relu(h_fc2_drop) 
        h_fc2_batn = batch_norm(h_fc2_lere)
        ## fc3 layer ##
        W_fc3 = weight_variable([1024, 256])
        b_fc3 = bias_variable([256])
        h_fc3 = tf.matmul(h_fc2_batn, W_fc3) + b_fc3
        h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)        
        h_fc3_lere = tf.nn.leaky_relu(h_fc3_drop) 
        h_fc3_batn = batch_norm(h_fc3_lere)
        # fc4 layer ##        
        W_fc4 = weight_variable([256, 1])
        b_fc4 = bias_variable([1])
        dis_out = tf.reduce_mean(tf.nn.sigmoid(tf.matmul(h_fc3_batn, W_fc4) + b_fc4))
        return dis_out


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, num_pixels])   # 28x28
ys = tf.placeholder(tf.float32, [None, num_pixels])
ps = tf.placeholder(tf.float32, [None, num_pixels])
x_image = tf.reshape(xs, [-1, TI, TI, 1])
y_image = tf.reshape(ys, [-1, TI, TI, 1])
p_image = tf.reshape(ps, [-1, TI, TI, 1])

keep_prob = tf.placeholder(tf.float32)
prediction = generator(x_image, keep_prob, reuse=False, name="generator")

dis_real = discriminator(x_image, y_image, keep_prob, reuse=False, name="discriminator_real")
dis_fake = discriminator(x_image, p_image, keep_prob, reuse=True, name="discriminator_fake")
dis_loss = tf.reduce_mean(-(tf.log(dis_real + EPS) + tf.log(1- dis_fake + EPS))) # 计算判别器的loss
gen_loss_l1 = tf.reduce_mean(tf.log(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1])))##计算平均误差，reduction_indices表示求和时函数的处理维度
gen_loss_gan = tf.reduce_mean(- tf.log( dis_fake + EPS)) 
gen_loss = gen_loss_gan  + 2*gen_loss_l1  # 计算生成器的loss

train_step = tf.train.AdamOptimizer(1e-4).minimize(gen_loss_l1)
g_train_step = tf.train.AdamOptimizer(1e-4).minimize(gen_loss)
d_train_step = tf.train.AdamOptimizer(1e-4).minimize(dis_loss)


# 开始运行程序
sess = tf.Session()
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

sum_epoch = 100        # 5    shixian6
sum_step_gen = 30   # 200    shixian30 !!!!写论文：自适应的次数
sum_step_dis = 100   # 5     shixian5   !!!!写论文：自适应的次数
sum_batch = 140      # sum_batch*len_batch_pat2tri<=22000
len_batch_pat2tri = 100 

train_dis_fake_gen = np.zeros((sum_epoch , sum_step_gen))
train_loss_gan = np.zeros((sum_epoch , sum_step_gen))
train_loss_l1 = np.zeros((sum_epoch , sum_step_gen))
train_loss = np.zeros((sum_epoch , sum_step_gen))
test_loss = np.zeros((sum_epoch , sum_step_gen))

train_dis_real = np.zeros((sum_epoch , sum_step_dis))
train_dis_fake = np.zeros((sum_epoch , sum_step_dis))
train_dis_loss = np.zeros((sum_epoch , sum_step_dis))

counter = 0

filename = "./Cnngan2020_Cyt2/"
if not os.path.exists(filename): #如果保存训练中可视化输出的文件夹不存在则创建
    os.makedirs(filename)
    
time_train_start=time.time()

for epoch in range(sum_epoch):
    
    time_trainG_start=time.time()

    # 训练G(generator)
    print(' ')
    print('epoch :', epoch+1)    
    for step in range(sum_step_gen):       
        for ii in range(sum_batch):  
            
            ii1_pat2tri = len_batch_pat2tri * ii
            ii2_pat2tri = len_batch_pat2tri * (ii + 1)  
            
            batch_xs_pat2tri = zl_sbp_pat2tritrain[ii1_pat2tri:ii2_pat2tri]
            batch_ys_pat2tri = Zl_pat2tritrain_imshow[ii1_pat2tri:ii2_pat2tri]
            feed_dict_pat2tri =  { xs: batch_xs_pat2tri, ys: batch_ys_pat2tri, keep_prob: 0.5 }           
            batch_ps_pat2tri = sess.run(prediction, feed_dict = feed_dict_pat2tri)
            feed_dict_gen_pat2tri =  { xs: batch_xs_pat2tri, ys: batch_ys_pat2tri, ps: batch_ps_pat2tri, keep_prob: 0.5 }  
            
            batch_xs_pat2tritest = zl_sbp_pat2tritest[1:500]
            batch_ys_pat2tritest = Zl_pat2tritest_imshow[1:500]
            feed_dict_test_pat2tri  = { xs: batch_xs_pat2tritest , ys: batch_ys_pat2tritest, keep_prob: 1} 
            batch_ps_pat2tri_test = sess.run(prediction, feed_dict = feed_dict_test_pat2tri)
            feed_dict_gen_pat2tri_test =  { xs: batch_xs_pat2tritest , ys: batch_ys_pat2tritest, ps: batch_ps_pat2tri_test, keep_prob: 1 }
            
            if epoch == 0:          # 按照CNN计算
                if ii == 0:
                    train_loss_l1[epoch,step]= sess.run(gen_loss_l1, feed_dict = feed_dict_pat2tri)
                    train_loss[epoch,step]= sess.run(gen_loss_l1, feed_dict = feed_dict_pat2tri)
                    test_loss[epoch,step]= sess.run(gen_loss_l1, feed_dict = feed_dict_test_pat2tri)
                sess.run(train_step, feed_dict = feed_dict_pat2tri)
            else:                   # 按照CNNCGAN计算
                if ii == 0:
                    train_dis_fake_gen[epoch,step], train_loss_gan[epoch,step], train_loss_l1[epoch,step], train_loss[epoch,step] = sess.run(
                            [dis_fake, gen_loss_gan, gen_loss_l1, gen_loss], feed_dict = feed_dict_gen_pat2tri)
                    test_loss[epoch,step]= sess.run(gen_loss, feed_dict = feed_dict_gen_pat2tri_test)
                sess.run(g_train_step, feed_dict = feed_dict_gen_pat2tri)
                                
        if (step+1) % 10 == 0: 
            print('  step_gen :', step+1)             
            # 训练集误差输出
            print('    train_loss= {:.3f}'.format(train_loss[epoch,step]))
            if epoch != 0:
                print('      dis_fake_gen= {:.3f} ,  loss_l1 = {:.3f}'.format(
                        train_dis_fake_gen[epoch,step],  train_loss_l1[epoch,step]))
            # 测试集误差计算及输出
            print('    test_loss = {:.3f}\n'.format(test_loss[epoch,step])) 

            
        if step == (sum_step_gen-1):
            # 训练集输出zs            
            feed_dict_out_pat2tri = { xs: zl_sbp_pat2tritrain, ys: Zl_pat2tritrain_imshow, keep_prob: 1 } 
            train_output_end_pat2tri  = sess.run(prediction, feed_dict = feed_dict_out_pat2tri )            
            # 测试输出zs
            feed_dict_pat2tritest = { xs: zl_sbp_pat2tritest, ys: Zl_pat2tritest_imshow, keep_prob: 1 } 
            test_output_end_pat2tri = sess.run(prediction, feed_dict = feed_dict_pat2tritest)
            print('end_gen\n')  

    time_trainG_end = time.time()            
    time_costG = time_trainG_end - time_trainG_start
    
    if (epoch+1) % 10 == 0:    
        # 实验反演结果
        print('epoch_shiyan:', epoch+1)
        feed_dict_shiyan = { xs: zl_sbp_shiyan, keep_prob: 1 } 
        shiyan_output_end = sess.run(prediction, feed_dict = feed_dict_shiyan)
        
        feed_dict_coms = { xs: zl_sbp_coms, keep_prob: 1 } 
        coms_output_end = sess.run(prediction, feed_dict = feed_dict_coms)
        
        write_image_name = filename + "shiyan_output_end" + str(epoch) + ".csv" 
        with open(write_image_name,'w',newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerows(shiyan_output_end)
            
        write_image_name = filename + "coms_output_end" + str(epoch) + ".csv" 
        with open(write_image_name,'w',newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerows(coms_output_end)
        print(' ')
    
        # 输出存储
        write_image_name = filename + "train_out" + str(epoch) + ".csv" 
        with open(write_image_name,'w',newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerows(train_output_end_pat2tri)
        write_image_name = filename + "test_out" + str(epoch) + ".csv" 
        with open(write_image_name,'w',newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerows(test_output_end_pat2tri)
        print(' ')
    
        # 误差存储    
        write_image_name = filename + "train_loss_gan" + ".csv"  
        with open(write_image_name,'w',newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerows(train_loss_gan)
        write_image_name = filename + "train_loss_l1" + ".csv"  
        with open(write_image_name,'w',newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerows(train_loss_l1)
        write_image_name = filename + "train_dis_fake_gen" + ".csv"  
        with open(write_image_name,'w',newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerows(train_dis_fake_gen)
        write_image_name = filename + "train_loss" + ".csv"  
        with open(write_image_name,'w',newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerows(train_loss)    
        write_image_name = filename + "test_loss" + ".csv" 
        with open(write_image_name,'w',newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerows(test_loss)
    
    time_trainD_start = time.time()            
    # 训练D(discriminator)
    print('dis')
    for step in range(sum_step_dis):       
        for ii in range(sum_batch):
            
            ii1_pat2tri = len_batch_pat2tri * ii
            ii2_pat2tri = len_batch_pat2tri * (ii + 1)

            batch_xs_pat2tri = zl_sbp_pat2tritrain[ii1_pat2tri:ii2_pat2tri]
            batch_ys_pat2tri = Zl_pat2tritrain_imshow[ii1_pat2tri:ii2_pat2tri]
            batch_ps_pat2tri = train_output_end_pat2tri[ii1_pat2tri:ii2_pat2tri]
            feed_dict_dis_pat2tri = { xs: batch_xs_pat2tri, ys: batch_ys_pat2tri, ps:batch_ps_pat2tri, keep_prob: 0.5 } 

            if ii == 0:
                train_dis_loss[epoch,step] , train_dis_real[epoch,step] , train_dis_fake[epoch,step] = sess.run(
                        [dis_loss, dis_real, dis_fake], feed_dict = feed_dict_dis_pat2tri)
            sess.run(d_train_step, feed_dict=feed_dict_dis_pat2tri)            

        if (step+1) % 10 == 0:                      
            print('  step_dis :', step+1)    
            print('    dis_loss = {:.3f} \n      dis_real = {:.3f},  dis_fake = {:.3f}'.format(
                    train_dis_loss[epoch,step] , train_dis_real[epoch,step] , train_dis_fake[epoch,step]))

        if train_dis_loss[epoch,step]<EPS2: 
            print('break_dis\n') 
            break

    time_trainD_end = time.time()            
    time_costD = time_trainD_end - time_trainD_start   
             
    counter += 2  
        
    if (epoch+1) % 10 == 0:    
        write_image_name = filename + "train_dis_loss" + ".csv"  
        with open(write_image_name,'w',newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerows(train_dis_loss)
        write_image_name = filename + "train_dis_real" + ".csv"  
        with open(write_image_name,'w',newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerows(train_dis_real)        
        write_image_name = filename + "train_dis_fake" + ".csv"  
        with open(write_image_name,'w',newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerows(train_dis_fake)        
    
time_train_end = time.time()            
time_cost = time_train_end - time_train_start
#
