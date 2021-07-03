# -*- coding: utf-8 -*-
"""
Created on Mon May 24 23:31:33 2021

@author: fatih
"""
import numpy as np
import matplotlib.pyplot as plt

train_data = np.array([[6,148,72,35,0,33.6,0.627,50],[1,85,66,29,0,26.6,0.351,31],[8,183,64,0,0,23.3,0.672,32],[1,89,66,23,94,28.1,0.167,21]])
train_data_target = np.array([[1],[0],[1],[0]])
input_neuron_size=train_data.shape[1]
hidden_neuron_size=3
output_neuron_size=train_data_target.shape[1]

train_data_size=train_data.shape[0]

weights1=np.random.rand(input_neuron_size,hidden_neuron_size)
weights2=np.random.rand(hidden_neuron_size,output_neuron_size)

eta=0.7
epoch=200

error_histoy=[]

def sigmoid(x):
    return 1/(1+np.exp(-x))

def predict(x):
    out_inputs=x
    
    out_hiddens=np.zeros(hidden_neuron_size)
    for i in range(input_neuron_size):
        for j in range(hidden_neuron_size):
            out_hiddens[j]=out_hiddens[j]+out_inputs[i]*weights1[i][j]
    
    for i in range(hidden_neuron_size):
        out_hiddens[i]=sigmoid(out_hiddens[i])
        
    output_outputs=np.zeros(output_neuron_size)
    for i in range(hidden_neuron_size):
        for j in range(output_neuron_size):
            output_outputs[j]=output_outputs[j]+out_hiddens[i]*weights2[i][j]
    
    for i in range(output_neuron_size):
        output_outputs[i]=sigmoid(output_outputs[i])
    
    return output_outputs

for iter in range(epoch):
    total_error=0
    for train_data_i in range(train_data_size):
        out_inputs=train_data[train_data_i,:]
        
        out_hiddens=np.zeros(hidden_neuron_size)
        for i in range(input_neuron_size):
            for j in range(hidden_neuron_size):
                out_hiddens[j]=out_hiddens[j]+out_inputs[i]*weights1[i][j]
        
        for i in range(hidden_neuron_size):
            out_hiddens[i]=sigmoid(out_hiddens[i])
        
        output_outputs=np.zeros(output_neuron_size)
        for i in range(hidden_neuron_size):
            for j in range(output_neuron_size):
                output_outputs[j]=output_outputs[j]+out_hiddens[i]*weights2[i][j]
        
        for i in range(output_neuron_size):
            output_outputs[i]=sigmoid(output_outputs[i])
        
        delta_outputs=np.zeros(output_neuron_size)
        for i in range(output_neuron_size):
            delta_outputs[i]=(train_data_target[train_data_i][i]-output_outputs[i])*output_outputs[i]*(1-output_outputs[i])
        
        for i in range(hidden_neuron_size):
            for j in range(output_neuron_size):
                weights2[i][j]=weights2[i][j]+eta*delta_outputs[j]*out_hiddens[i]
        
        delta_hiddens=np.zeros(hidden_neuron_size)
        for i in range(hidden_neuron_size):
            total_sum=0
            for j in range(output_neuron_size):
                total_sum=total_sum+(weights2[i][j]*delta_outputs[j])
            delta_hiddens[i]=out_hiddens[i]*(1-out_hiddens[i])*total_sum
        
        for i in range(input_neuron_size):
            for j in range(hidden_neuron_size):
                weights1[i][j]=weights1[i][j]+eta*delta_hiddens[j]*out_inputs[i]
        
        total_error=total_error+np.sum(np.power(train_data_target[train_data_i]-output_outputs,2))
    
    error_histoy.append(total_error)
    print("epoh: ",iter,"\tError: ",total_error)
    
plt.figure(figsize=(5,5))
plt.plot(error_histoy)
plt.xlabel("Epoch")
plt.ylabel("Hata")
plt.show()