from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

import pandas as pd
import numpy as np
# Sample data
data = ['cat', 'dog', 'cat', 'mouse', 'dog']
# Label Encoding
label_encoder = LabelEncoder()
labeled_data = label_encoder.fit_transform(data)    
print("Label Encoded Data:", labeled_data)
# One-Hot Encoding

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(np.array(data).reshape(-1, 1))
print("One-Hot Encoded Data:\n", encoded)


# Ordinal Encoding
ordinal_encoder = OrdinalEncoder()
ordinal_encoded = ordinal_encoder.fit_transform(np.array(data).reshape(-1, 1))
print("Ordinal Encoded Data:\n", ordinal_encoded)

#Frequency encoding  == Tf-idf encoding

#sent1 = "I love machine learning and learning is fun"    i fun, love, machine learning
#sent2 = "I enjoy learning new things and machine learning is great"

# TF =   learning =  2 
# IDF = log( 2/2) = 0 
# TF-IDF = TF * IDF = 0

# machine in sentence 1 = 1   
# IDF = log(2/2) = 0

#  fun = 1
# IDF = log(2/1) = 0.693
# TF-IDF = 1 * 0.693 = 0.693

#  0 to 1 

# Word pattern -- us ( frequency  of word, which context it is used in ) -- TF-IDF -- 0 to 1


# Transform

# 1. self-attention mechanism
# 2. posining encoding 

# I(1) love( 2) machine learning and learning is fun ( what is the import of the word i with respect to other words in that sentence)


#  machine learning 

#  vectorization -- converting the word into number with context and meaning  [ ]


#  encoding - simple numerical representation
#  vectorization - dense vector  or sparse vector representation
#  embedding - dense semantic vector representation   


#  how to learning python ? - 0 
#  how to study python ?  - 2 

# <--learning -> studying -> knowledge - > understanding








# to position which place the word has to come transformers came into the picture

# issues in encoding - Transformers came into the picture 
# 1.self attention 
# 2.position encoding

# I(1) love(2) machine learning and learning is fun (what is the importance of word i with respect to other word )

# encoding - converting text into number 
# embedding & vectorization - works on top of encoding, 3D of representing word in 1D number into 3D vector format 
# indepth of understanding the word -

# vectorization - convert the word into number with context and with meaning (array format) without word being present in sentence  without understanding the transformations of words or equivalant word (learning--machine learning , understanding context of  word ) 
# encoding - converting word in to number without meaning 
# embedding - converting the word into number with context and meaning and dense semantic representation of word in the sentence (transformation - leanrning - learn , learned, learning , studying - suitable words and mapping of words, equivalent words su )




