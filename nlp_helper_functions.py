

"""
Helper functions used for several preprocessing on json dataset for 
natural language processing task in question answering by text extraction.
"""

### to reduce the long installation outputs
from IPython.display import clear_output


## importing dependencies
import json
from pathlib import Path
import zipfile
import re

import tensorflow as tf
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

## word embedding model
import gensim

## filtering out warnings
import warnings
warnings.filterwarnings('ignore')

###################################################################

def read_fquad(path):
	 """
	 Give the path of the current directory of the json file.
	 The function returns `context`, `questions`, and `answers` in a list format from the json document.
	 """
	path = Path(path)
	with open(path, 'rb') as f:
	   squad_dict = json.load(f)

	contexts = []
	questions = []
	answers = []
	for group in squad_dict['data']:
	    for passage in group['paragraphs']:
	        context = passage['context']
	         for qa in passage['qas']:
	            question = qa['question']
	            for answer in qa['answers']:
	                contexts.append(context)
	                questions.append(question)
	                answers.append(answer)

	return contexts, questions, answers

#################################################################
def add_end_idx(answers, contexts):

  	"""
  	answers = A dict object with `answer_start` and `text` 
  	The functions fixes the mixed placed word indexes in the answers.
  	Relative to the given context
  	"""
  	for answer, context in zip(answers, contexts):
	    gold_text = answer['text']
	    start_idx = answer['answer_start']
	    end_idx = start_idx + len(gold_text.split())
	    answer['answer_end'] = int(end_idx)

	    # sometimes the answers are off by a character or two – we fix this 
    if context[start_idx:end_idx] == gold_text:
        answer['answer_end'] = end_idx
    elif context[start_idx-1:end_idx-1] == gold_text:
        answer['answer_start'] = start_idx - 1
        answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        
    elif context[start_idx-2:end_idx-2] == gold_text:
        answer['answer_start'] = start_idx - 2
        answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

    elif context[start_idx-3:end_idx-3] == gold_text:
      answer['answer_start'] = start_idx - 3
      answer['answer_end'] = end_idx - 3		# When the gold text is off by three characters

    elif context[start_idx-4:end_idx-4] == gold_text:
      answer['answer_start'] = start_idx - 4
      answer['answer_end'] = end_idx - 4 		# When the gold text is off by four characters

    elif context[start_idx-5:end_idx-5] == gold_text:
      answer['answer_start'] = start_idx - 5
      answer['answer_end'] = add_end_idx - 5		# When the gold text is off by five characters

      ############################################################################

def build_dataframe(train_contexts, 
                    train_questions, 
                    train_answers
                    ):
  	"""
  	The function builds a dataframe for the output of the read_fquad method.
  	"""
  	samples = []
  	for i in range(len(train_contexts)):
    	context = train_contexts[i]
    	question = train_questions[i]
    	answer = train_answers[i]['text']
    	ans_start = train_answers[i]['answer_start']
    	ans_end = train_answers[i]['answer_end']

    	samples.append([context, question, answer, ans_start, ans_end])

  	df = pd.DataFrame(samples, columns = ['context', 'questions', 'answers', 'answer_start', 'answer_end'])
  	return df

##################################################################
def lower(df):

  	"""
  	The dataframe must contain the followinf columns; context, questions, answers
  	This fucntion lower all upper cases in the text columns
  	"""
  	df['context'] = [i.lower() for i in df.context]
  	df['questions'] = [i.lower() for i in df.questions]
  	df['answers'] = [i.lower() for i in df.answers]

  	return df

#############################################################

def fix_answer_index(df):

  	"""
  	Due to spotted positioning of answer tokens, this method generates 
  	the right positioning of the end and start tokens for the answers
  	Pass in the dataframe with the `context`, `answers` tokens to fix it. 
  	"""
  	for i in range(len(df)):

    	context_tokens = re.split(",| |_|-|!|\.|\(|\)|’|=|'", df.context[i])
    	answer_tokens = re.split(",| |_|-|!|\.|\(|\)|’|=|'", df.answers[i])
    	#print(i)
    	for ans_token in context_tokens:
      		ans_start = int(context_tokens.index(answer_tokens[0])) 
      		ans_end = int(ans_start + len(answer_tokens))

      		df['answer_start'].loc[i] = ans_start
      		df['answer_end'].loc[i] = ans_end
  	return df

##########################################################
