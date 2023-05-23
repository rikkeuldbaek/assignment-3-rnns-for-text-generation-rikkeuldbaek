### Language Assignment 3
## Cultural Data Science - Language Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 28th of March 2023

#--------------------------------------------------------#
################## TRANING THE MODEL #####################
#--------------------------------------------------------#

# (please note that some of this code has been adapted from class sessions)

# data processing tools 
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

# Scripting
import argparse

###################### PARSER ############################
def input_parse():
    #initialise the parser
    parser = argparse.ArgumentParser()

    #add arguments arguments for train_model.py
    parser.add_argument("--n_epochs", type=int, default= 5, help= "Specify number of epochs. More epochs increase accuracy but also computational time of running.") 
    parser.add_argument("--batch_size", type=int, default= 40, help= "Specify size of batch size. The batch size refers to the number of samples which are propagated through the network.")
    parser.add_argument("--verbose", type=int, default= 1, help= "Specify whether the training progress for each epoch should be displayed.") 
    
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

# Import data
import data as dt

args = input_parse()
all_comments = dt.data_load(args.member_folder, args.sub_folder, args.word_in_filename, args.column_name)
corpus = dt.data_clean(all_comments, args.n_comments)
tokenizer, total_words = dt.data_tokenize(corpus)
predictors, label, max_sequence_len, total_words, tokenizer = dt.data_seq_and_pad(tokenizer, total_words, corpus)


##################### TRAIN THE MODEL #####################

# Create model
def model_func(max_sequence_len, total_words, predictors, label, n_epochs, batch_size, verbose):
    # Contruct model
    model = pdfunc.create_model(max_sequence_len, total_words) 

    # Create history
    history = model.fit(predictors,
                        label, 
                        epochs= n_epochs, 
                        batch_size= batch_size, 
                        verbose=1) 

    return history, model

print(max_sequence_len)


############# SAVE THE MODEL, TOKENIZER & MAX_SEQ_LENGTH #############
def save_func(history, model, tokenizer, max_sequence_len):
    file_path = os.path.join(os.getcwd(),"models") 
    tf.keras.models.save_model(
        model,
        file_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True
    ) 
    
    #save tokenizer
    outpath_tokenizer = os.path.join(os.getcwd(), "out", "tokenizer.pickle")
    dump(tokenizer, open(outpath_tokenizer, 'wb'))
    
    #save max_sequence_len
    f = open("out/max_sequence_len.txt", "w")
    f.write(str(max_sequence_len))
    f.close()



#################### MAIN FUNCTION #######################
def main():
    args = input_parse()
    print("Initializing training of model..")
    history, model = model_func(max_sequence_len, total_words, predictors, label, args.n_epochs, args.batch_size, args.verbose)
    print("Training done!")
    save_func(history, model, tokenizer, max_sequence_len)
    print("Saving model and metrics..")

if __name__ == '__main__':
    main()