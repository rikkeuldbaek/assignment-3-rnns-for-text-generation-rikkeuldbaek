### Language Assignment 3
## Cultural Data Science - Language Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 28th of March 2023

#--------------------------------------------------------#
################## DATA PREPROCESSING ####################
#--------------------------------------------------------#

# (please note that some of this code has been adapted from class sessions)

# data processing tools
import string, os, sys
import pandas as pd
import numpy as np
np.random.seed(42)
#from random import sample
import random

# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.preprocessing.text import Tokenizer

# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import predefined functions 
sys.path.append(os.path.join("utils"))
import predef_func as pdfunc


# Scripting
import argparse

###################### PARSER ############################
def input_parse():
    #initialise the parser
    parser = argparse.ArgumentParser()

    #add arguments for data.py
    parser.add_argument("--member_folder", type=str, default= "431868", help= "Specify your specific member folder where the data is located.") 
    parser.add_argument("--sub_folder", type=str, default= "news_data", help= "Specify the subfolder of the member folder where the .csv files are located.") 
    parser.add_argument("--word_in_filename", type=str, default= "Comments", help= "Specify a given word in filenames, to loop over files of same type.") 
    parser.add_argument("--column_name", type=str, default= "commentBody", help= "Specify name of column that contains the necessary text for modelling.") 
    parser.add_argument("--n_comments", type=int, default= 2000, help= "Specify the amount of comments that are randomly sampled.") 


    # parse the arguments from the command line 
    args = parser.parse_args()
    
    #define a return value
    return args #returning arguments



#################### DATA PREPROCESSING ####################

# Load NY times Comments.csv files
def data_load(member_folder, sub_folder, word_in_filename, column_name):
    data_dir = os.path.join("..", "..", "..", member_folder, sub_folder)  

    # Append only the comments to the list of data in a loop
    all_comments = []

    for file in os.listdir(data_dir):
        if word_in_filename in file:
            comments_df = pd.read_csv(data_dir +"/"+ file)
            all_comments.extend(list(comments_df[column_name].values)) # keep all comments
    return all_comments


# Clean data
def data_clean(all_comments, n_comments):
    
    #sample comments
    sample_comments = all_comments[:n_comments]
    print("Using " + str(n_comments) + " comments.")
    sample_comments = [c for c in sample_comments if c != "Unknown"] #if the comment is unknown, remove it
    corpus = [pdfunc.clean_text(x) for x in sample_comments] # clean the text corpora 
    return corpus


# Tokenize the data
def data_tokenize(corpus):
    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    return tokenizer, total_words


## Get sequence of tokens and pad the sequences
def data_seq_and_pad(tokenizer, total_words, corpus):
    # Create input sequences of tokens
    inp_sequences = pdfunc.get_sequence_of_tokens(tokenizer, corpus)
    # Overcome fixed dimensionality and pad the sequences with 0's until of same length
    predictors, label, max_sequence_len = pdfunc.generate_padded_sequences(inp_sequences, total_words) 
    return predictors, label, max_sequence_len, total_words, tokenizer 


#################### MAIN FUNCTION #######################
def main():
    args = input_parse()
    print("Initializing data preprocessing..")
    all_comments = data_load(args.member_folder, args.sub_folder, args.word_in_filename, args.column_name)
    corpus = data_clean(all_comments, args.n_comments)
    tokenizer, total_words = data_tokenize(corpus)
    predictors, label, max_sequence_len, total_words, tokenizer = data_seq_and_pad(tokenizer, total_words, corpus)
    print("Data preprocessing done!")

if __name__ == '__main__':
    main()
