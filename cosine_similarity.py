#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 16:03:08 2021

@author: ehab
"""
import tensorflow as tf
        
def _similarity(self, a, b):
    if self.similarity_type =='cosine':
        a = tf.nn.l2_normalize(a, -1)
        b = tf.nn.l2_normalize(b, -1)
        
        
    if self.similarity_type =='cosine' or self.similarity_type == 'inner':
        sim = tf.reduce_sum(tf.expand_dims(a,1)*b, -1)
        
        #simiilarity between intent embeddings
        sim_emb = tf.reduce_sum(b[:, 0:1, :]*b[:, 1:, :],-1)
        return sim, sim_emb
    
    
    
    else:
        raise ValueError("Wrong similarity type {}, ""should be 'cosine' or 'inner'""".format(self.similarity_type))
        