[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-8d59dc4de5201274e310e4c54b9627a8934c3b88527886e3b421487c677d23eb.svg)](https://classroom.github.com/a/5f7lMH9Y)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10586985&assignment_repo_type=AssignmentRepo)

# **Assignment 3 - Language modelling and text generation using RNNs**
## **Cultural Data Science - Language Analytics** 
#### Author: Rikke Uldbæk (202007501)
#### Date: 28th of March 2023
<br>


# **3.1 GitHub link**
The following link is a link to the GitHub repository of assignment 3 in the course Language Analytics (F23.147201U021.A). Within the GitHub repository all necessary code are provided to reproduce the results of the assignment.

https://github.com/rikkeuldbaek/assignment-3-rnns-for-text-generation-rikkeuldbaek

<br>

# **3.2 Description**
For this assignment, I have build a text-generation RNN model which has been trained on a large text corpora of *comments* on articles from *The New York Times dataset*. This trained model is able to generate text from a user-specific prompt as demonstrated further below. In order to build this complex deep learning model for NLP I have used the tools available via ```TensorFlow```. 

I have created a collection of scripts which do the following: Train a model on the Comments section of the data, save the trained model in the folder ```models```, load the saved model, and generate text from a user-suggested prompt.

<br>

# **3.3 Data**
The entire *New York Times dataset* consists of both *articles* published in the New York Times and *comments on articles*. For this assignment I will only be using the *comments on articles* in order to make text generation. The comments section of the articles is highly active and provides an insight to the readers' opinions on the commented articles. The .csv files of comments contain around 2 million comments, which has been collected from Jan-May 2017 and Jan-April 2018. The data is available via Kaggle, please see resources for further information. 

<br>


# **3.4 Repository Structure**
The scripts require a certain folder structure, thus the table below presents the required folders and their description and content.

|Folder name|Description|Content|
|---|---|---|
|```src```|model and data scripts|```data.py```, ```train_model.py```, ```run_model.py```|
|```models```| saved RNN model and it's parameters and variables|```saved_model.pb```, ```keras_metadata.pb```, ```fingerprint.pb``` etc.|
|```out```|tokenizer and max_sequence_len|```tokenizer.pickle```, ```max_sequence_len.txt```|
|```utils```|predefined functions|```predef_func.py```|
|```data```|storage of data for the user|currently empty, download data via kaggle and se further instructions below|


The ```data.py``` script located in ```src``` produces preprocessed, cleaned, tokenized, and padded data. The ```train_model.py``` located in ```src``` produces a trained model that are saved in the folder ```models``` and likewise saves the tokenizer and max-sequence-length variable for later use when running the model. Lastly, the ```run_model.py``` located in ```src``` produces a sequence of generated text from user-suggested input prompt. 


<br>

# **3.5 Usage and Reproducibility**
## **3.5.1 Prerequisites** 
In order for the user to be able to run the code, please make sure to have bash and python 3 installed on the used device. The code has been written and tested with Python 3.9.2 on a Linux operating system. In order to run the provided code for this assignment, please follow the instructions below.

<br>

## **3.5.2 Setup Instructions** 
**1) Clone the repository**
```python
git clone https://github.com/rikkeuldbaek/assignment-3-rnns-for-text-generation-rikkeuldbaek
 ```

 **2) Setup** <br>
Setup virtual environment (```LA3_env```) and install required packages.
```python
bash setup.sh
```
 
 **3) Download the data** <br>
Two options of data acquisition are available. The data must either be downloaded and stored inside the repository in the folder ```data``` or if using UCloud the ```cds-lang-data``` folder must be added to the run. 

 **3.1) Download the data on Kaggle** <br>
Download the data via kaggle (see link in resources), and place the comments .csv files in the ```data``` folder in the respository. Please change the path in line 61 in the ```data.py``` script in order to access the data. 


 **3.2) Add data folder on UCloud** <br>
The data is already stored in the ```cds-lang-data``` folder on UCloud, hence if accessible for the user please add this folder to the run. 


<br>

## **3.5.3 Run the script** 
In order to run the text generation RNN please execute the command below in the terminal. This command will *only* run the trained model and not re-train it. For the sake of the demonstration only, I have quoted out the ```train_model.py``` in the bash file so the user does not have to spend a lot of time training the model. If the user would like to re-train the model on different parameters, please unquote the ```train_model.py``` script in the ```run.sh``` file and change the available arguments. 

```python
bash run.sh
```

<br>

## **3.5.4 Script arguments**
In order to make user-specific modifications, the three scripts have the following arguments stated below. These arguments can be modified and adjusted in the ```run.sh``` script or when run individually. If no modifications is added, default parameters are used. In case help is needed, please write --help in continuation of the code below instead of writing an argument.


The ```data.py``` takes the following arguments:

|Argument|Type|Default|
|---|---|---|
|--member_folder|string |"431868" |
|--sub_folder |string |"news_data"|
|--word_in_filename |string |"Comments" |
|--column_name |string |"commentBody" |
|--n_comments |integer |2000 |


<br>

The ```train_model.py``` takes the following arguments:

|Argument|Type|Default|
|---|---|---|
|--n_epochs |integer |5 |
|--batch_size |integer |40 |
|--verbose |integer | 1|


<br>

The ```run_model.py``` takes the following arguments:

|Argument|Type|Default|
|---|---|---|
|--word |string |"Trump" |
|--n_next_words |integer |8 |


<br>

**Important to note** <br>
The ```data.py``` is automatically called upon when training the model in the ```train_model.py``` script, thus the arguments for ```data.py``` must be parsed in the ```train_model.py``` script inside the ```run.sh``` bash file:

````python 
python src/train_model.py --arguments_for_training --arguments_for_data
````


# **3.6 Application of the user-specific prompt**
In order to generate text from the *New York Times comments* the user has two arguments to modify: ```--word``` and ```--n_next_words```. The ```--word``` argument constitute the first word in the generated sentence, while the ```--n_next_words``` argument determines how many words should follow the ```--word```. As previously mentioned, these arguments must be added in the ```run.sh``` script. A thought example could be: 

````python
python src/run_model.py --word Trump --n_next_words 8
````

Example of possible output:
````
Trump was the President of United States of America
````

<br>


# **3.7 Results**
The dataset of comments from *The New York Times* is very large and this causes some computational problems when preprocessing the data. Thus, the RNN will not be trained on the entire dataset but on a subset of the data. The current default of the sample size is the first 2000 comments, however this can be modified with the argument ```--n_comments```. The provided code will in theory work on the entire dataset, but in practice it is not possible due to the computational limitations. Using a rather small subset of the data when training the model influences the model performance a lot. Thus, the user should expect the model to perform quite poor and the text that the model is supposed to generate will probably not make much sense. This assignment's trained RNN model with the default word prompt "Trump" and default 8 next words to follow generates the following sentence:

````
Trump Is A Lot Of The Hall Of Fame
````

This generated sentence is gramatically and semantically quite poor. However, the quality of the generated sentence should increase if the RNN is trained on more data. 


<br>

# **Resources**

[The New York Times data](https://www.kaggle.com/datasets/aashita/nyt-comments)
(https://www.kaggle.com/datasets/aashita/nyt-comments)

[TensorFlow](https://www.tensorflow.org/)

[TensorFlow - keras module](https://www.tensorflow.org/api_docs/python/tf/keras)
