from __future__ import print_function
import argparse
import pickle
import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import distance
import sys
import os
import scipy
import random
cwd = os.getcwd()
sys.path.append(cwd+'/../')

#
# this part is about the algorithm 3
# authors uses the cos distance to judge similarity between W and the newly generated w*
# besides it also uses the batch normalization
#

def create_scaling_mat_ip_thres_bias(weight, ind, th, m_type):

    cosine_sim = 1-pairwise_distances(weight, metric="cosine")
    weight_chosen = weight[ind, :]

    #
    # the goal of this matrix is to mark which element in
    # and make the weight and weight_chosen into one list
    #
    scaling_mat = np.zeros([weight.shape[0], weight_chosen.shape[0]])

    #print(scaling_mat)

    for i in range(weight.shape[0]):
        if i in ind: # chosen
            ind_i, = np.where(ind == i)
            scaling_mat[i, ind_i] = 1
        else:
            #
            # if the model type is 'prune',we do not need to
            # consider the element outside ind, since they are cut
            #
            if m_type == 'prune':
                continue
            #
            # in this step, I find the maximum element in cosine_sim
            # since in the range of 0 to π the cosine function is monotomic
            # decreasing, the more similar the two vectors are,
            # the closer the cos value approaches 1(larger than others).
            # Therefore, if we use sort function to realize it, it should be the
            # last elementr
            #
            max_cos_value = np.sort(cosine_sim[i][ind])[-1]
            #
            # find the index of maximum cosine distance from cosine_sim[i][ind]
            #

            max_cos_value_index = np.argpartition(cosine_sim[i][ind], -1)[-1]


            # if cos value is smaller than the threhold
            # and th is not zero, we do not need to
            # update the scaling matrix, therefore we should
            # skip next part

            if th and max_cos_value < th:
                continue

            # this part is updating the scaling matrix,
            # the main idea is preparing current with baseline

            #
            # use number index of weight_chosen to be the basline weight
            # and now we loop number is ｉ　，so we should also take
            # weight[i] as the current weight
            #
            # figure out the norm of the baseline weight and current weight
            #
            baseline_norm = np.linalg.norm(weight_chosen[max_cos_value_index])
            current_norm = np.linalg.norm(weight[i])
            #
            # use the current norm and baseline norm to update the scaling matrix
            #
            scaling_mat[i, max_cos_value_index] = current_norm / baseline_norm

    return scaling_mat

#
# th is about the cosine similarity
# bn_weight and bn_bias are parameters of norm after convolution
# bn_mean and bn_var are means and vars of BN
# lam is the cosine distance
#
# this part is about to create scaling matrix convolution with
# batch normalization layer
#
def create_scaling_mat_conv_thres_bn(weight, ind, th,
                                     bn_weight, bn_bias,
                                     bn_mean, bn_var, lam, m_type):
    
    weight = weight.reshape(weight.shape[0], -1)
    #
    # we can use function named pairwise_distance in sklearn.metrics package
    #
    cosine_dist = pairwise_distances(weight, metric="cosine")

    weight_chosen = weight[ind, :]
    scaling_mat = np.zeros([weight.shape[0], weight_chosen.shape[0]])
    #
    # this part is mainly about batch normalization layer
    # loop the weight matrix like before
    # To be more specific, this part realized the formula 8 though
    # the iterations
    #
    for i in range(weight.shape[0]):
        if i in ind:
            ind_i, = np.where(ind == i)
            #
            # mark the data in ind set
            #
            scaling_mat[i, ind_i] = 1
        else:
            #
            # this part is the model is prune and this element
            # is not in the chosen set(ind), hence it does not need
            # to compute the parameters
            #
            if m_type == 'prune':
                continue
            #
            # use formula 8 to figure out the parameters
            # current weight is about the weight[i]
            #
            current_norm = np.linalg.norm( weight[i])
            current_cos = cosine_dist[i]
            gamma_1 = bn_weight[i]
            beta_1 = bn_bias[i]
            mu_1 = bn_mean[i]
            sigma_1 = bn_var[i]
            
            # prepare the empty list
            cos_list = []
            scale_list = []
            bias_list = []


            for chosen_i in ind:
                #
                # loop the element in ind set again and figure out
                # the current parameters
                #
                chosen_norm = np.linalg.norm(weight[chosen_i], ord = 2)
                chosen_cos = current_cos[chosen_i]
                gamma_2 = bn_weight[chosen_i]
                beta_2 = bn_bias[chosen_i]
                mu_2 = bn_mean[chosen_i]
                sigma_2 = bn_var[chosen_i]
                
                # compute cosine sim
                cos_list.append(chosen_cos)
                
                # compute s
                s = current_norm/chosen_norm
                #
                # compute scale term
                #
                scale_list.append(s * (gamma_2 / gamma_1) * (sigma_1 / sigma_2))
                
                # compute bias term and add into the list

                bias_list.append( (abs((gamma_2/sigma_2) * (s * (-(sigma_1*beta_1/gamma_1) + mu_1) - mu_2) + beta_2))/(s * (gamma_2 / gamma_1) * (sigma_1 / sigma_2)))


            # after looping, we should merge cosine distance and bias distance

            bias_up = np.sort(bias_list)
            bias_list = (bias_list - bias_up[0]) / (bias_up[-1]-bias_up[0])

            score_list = lam * np.array(cos_list) + (1-lam) * np.array(bias_list)


            # find index and scale with minimum distance
            min_ind = np.argmin(score_list)

            min_scale = scale_list[min_ind]
            min_cosine_sim = 1-cos_list[min_ind]

            #
            # if min cos distribution less than threshold
            # we do not need to update the scaling matrix
            # since before result is better than this iteration result
            # we can use it directly
            #

            if th and min_cosine_sim < th:
                continue
            
            scaling_mat[i, min_ind] = min_scale

    return scaling_mat


class Decompose:
    def __init__(self, arch, p_dic, cri, th, lam, m_type, cfg, cuda):
        
        self.p_dic = p_dic
        self.arch = arch
        self.cri = cri
        self.th = th
        self.lam = lam
        self.m_type = m_type
        self.cfg = cfg
        self.cuda = cuda
        self.out_ch_index = {}
        self.dec_weight = []

    def get_out_ch_index(self, value, id):

        out_ch_index = []

        if len(value.size()) :

            weight_vec = value.view(value.size()[0], -1)
            weight_vec = weight_vec.cuda()

            # l1-norm
            if self.cri == 'l1-norm':
                norm = torch.norm(weight_vec, 1, 1)
                norm_np = norm.cpu().detach().numpy()
                arg_max = np.argsort(norm_np)
                arg_max_rev = arg_max[::-1][:self.cfg[id]]
                out_ch_index = sorted(arg_max_rev.tolist())
            
            # l2-norm
            elif self.cri == 'l2-norm':
                norm = torch.norm(weight_vec, 2, 1)
                norm_np = norm.cpu().detach().numpy()
                arg_max = np.argsort(norm_np)
                arg_max_rev = arg_max[::-1][:self.cfg[id]]
                out_ch_index = sorted(arg_max_rev.tolist())

        return out_ch_index




    def get_dec_weight(self):

        # scale matrix
        z = None

        # copy original weight
        self.dec_weight = list(self.p_dic.values())

        # cfg index
        id = -1

        for index, layer in enumerate(self.p_dic):

            a_f = self.p_dic[layer]

            # LeNet_300_100
            if self.arch == 'LeNet_300_100':

                # ip
                if layer in ['fc1.weight','fc2.weight'] :

                    #
                    # Merge scale matrix
                    #
                    # start with the second iteration
                    # since in the first step z is None, in this case, we should use initiation data
                    # and use these data to make merge scale matrix
                    # about how to use matrix multiplication to realize it
                    # since using the last step data to compute the matrix by matrix a and matrix f
                    # it is named a_f
                    #
                    #
                    if z != None:
                        a_f = a_f@z
                    #
                    # the order of layer adds one
                    #
                    id += 1

                    # concatenate weight and bias
                    # use the function of np.concatenate to link the weight and
                    # bias, and then use it to make the information of information
                    # supplement matrix of z
                    if layer in 'fc1.weight' :
                        weight = self.p_dic['fc1.weight'].cpu().detach().numpy()
                        bias = self.p_dic['fc1.bias'].cpu().detach().numpy()

                    elif layer in 'fc2.weight' :
                        weight = self.p_dic['fc2.weight'].cpu().detach().numpy()
                        bias = self.p_dic['fc2.bias'].cpu().detach().numpy()

                    bias_reshaped = bias.reshape(bias.shape[0],-1)
                    concat_weight = np.concatenate([weight, bias_reshaped], axis = 1)

                    
                    # get index
                    self.out_ch_index[index] = self.get_out_ch_index(torch.from_numpy(concat_weight), id)

                    #
                    # make scale matrix with bias
                    # as we all know, the create_scaling_mat_ip_thres_bias function is
                    # the similarity of matrics, in this paper, it as the information
                    # supplement matrix
                    #
                    # pay more attention
                    # we should change the type of data
                    #
                    z = torch.from_numpy(create_scaling_mat_ip_thres_bias(concat_weight, np.array(self.out_ch_index[index]), self.th, self.m_type)).type(dtype=torch.float)

                    #
                    # put z onto GPU
                    #
                    if self.cuda:
                        z = z.cuda()
                    

                    # pruned the a_f matrix in the paper
                    pruned = a_f[self.out_ch_index[index],:]

                    # update next input channel
                    input_channel_index = self.out_ch_index[index]

                    # update decompose weight
                    self.dec_weight[index] = pruned

                elif layer in 'fc3.weight':

                    a_f = torch.mm(a_f,z)

                    # update decompose weight
                    self.dec_weight[index] = a_f

                # update bias
                elif layer in ['fc1.bias','fc2.bias']:
                    self.dec_weight[index] = a_f[input_channel_index]
                
                else :
                    pass                    
                    

    def main(self):

        if self.cuda == False:
            for layer in self.p_dic:
                self.p_dic[layer] = self.p_dic[layer].cpu()

        self.get_dec_weight()

        return self.dec_weight