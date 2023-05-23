### Language Assignment 3
## Cultural Data Science - Language Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 28th of March 2023

#--------------------------------------------------------#
################## RUNNING THE MODEL #####################
#--------------------------------------------------------#

# (please note that some of this code has been adapted from class sessions)
 
# data processing tools
# Scripting
import argparse
import string, os, sys 
import pandas as pd
import numpy as np
np.random.seed(42)

# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)

# Save things
from joblib import dump
import pickle

# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import predefined functions 
sys.path.append(os.path.join("utils"))
import predef_func as pdfunc


###################### PARSER ############################
def input_parse():
    #initialise the parser
    parser = argparse.ArgumentParser()

    #add arguments for run_model.py
    parser.add_argument("--word", type=str, default= "Trump", help= "Specify word for text generation.") 
    parser.add_argument("--n_next_words", type=int, default= 8, help= "Specify number of next words following the chosen word.")

    # parse the arguments from the command line 
    args = parser.parse_args()
    
    #define a return value 
    return args #returning arguments


#################### IMPORT DATA ######################## 
# Import tokenizer and max_sequence_len
def load_the_data():
    # loading tokenizer
    with open('out/tokenizer.pickle', 'rb') as t:
        tokenizer = pickle.load(t)

    # loading the max_seq_len
    with open('out/max_sequence_len.txt') as f:
        max_sequence_len = f.read()
    
    # make max_sequence_len integer    
    max_sequence_len = int(max_sequence_len)

    return tokenizer, max_sequence_len

#################### RUN THE MODEL #######################
def run_model(word, n_next_words, max_sequence_len, tokenizer):
    #load model
    file_path = os.path.join(os.getcwd(),"models")
    loaded_model = tf.keras.models.load_model(file_path) 
    print(pdfunc.generate_text(word, n_next_words, loaded_model, max_sequence_len, tokenizer)) # Generated text 


#################### MAIN FUNCTION #######################
def main():
    args = input_parse()
    tokenizer, max_sequence_len = load_the_data()
    print("Generating text from the word: "+ args.word)
    run_model(args.word, args.n_next_words, max_sequence_len, tokenizer)

if __name__ == '__main__':
    main() 
