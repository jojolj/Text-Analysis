# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 19:16:55 2024

@author: Jin Li
"""
import numpy as np
def read_and_one_hot_encode(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    unique_chars = sorted(set(text))
    char_to_int = {char: idx for idx, char in enumerate(unique_chars)}
    int_to_char = {idx: char for idx, char in enumerate(unique_chars)}
    
    num_chars = len(text)
    vocab_size = len(unique_chars)
    
    one_hot_matrix = np.zeros((num_chars, vocab_size), dtype=int)
    
    for i, char in enumerate(text):
        one_hot_matrix[i, char_to_int[char]] = 1
    
    return one_hot_matrix, vocab_size, char_to_int, int_to_char, num_chars

# Assuming file path is correctly defined
file_path = 'D:\\MasterStudies\\PatternRecognitionandMachineLearning\\Exercises\\abcde.txt'
# file_path = 'D:\\MasterStudies\\PatternRecognitionandMachineLearning\\Exercises\\abcde_edcba.txt'
one_hot_data, vocab_size, char_to_int, int_to_char, total_chars = read_and_one_hot_encode(file_path)

print("one_hot_data:", one_hot_data)
print("Total number of characters:", total_chars)
print("Size of vocabulary:", vocab_size)
print("One-hot encoded data shape:", one_hot_data.shape)
#print("Unique characters:", unique_chars)