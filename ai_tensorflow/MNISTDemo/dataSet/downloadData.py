'''
Created on 2017年12月14日
download data for Mnist
@author: zhangcf17306
'''
import input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)