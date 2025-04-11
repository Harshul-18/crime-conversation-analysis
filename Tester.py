# pip install transformers
# pip install tensorflow
# pip install nltk


import string
import numpy as np
import pandas as pd
import os
import time
import nltk
import shutil
import warnings 
import sys
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=UserWarning, module='absl')
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from transformers import (
    BertTokenizer, 
    TFBertForSequenceClassification
)
from transformers import (
    InputExample, 
    InputFeatures
)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)

def expand_contractions(sentence, initial_case='uppercase'):
    contractions = {"ain't": "am not", 
                "aren't": "are not", 
                "can't": "cannot", 
                "can't've": "cannot have", "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so is", "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
    if initial_case == 'lowercase':
        contractions = {k.lower(): v.lower() for k, v in contractions.items()}
        sentence = sentence.lower()
    elif initial_case == 'uppercase':
        contractions = contractions
    else:
        print('''
** Invalid initial_case option. Choose from lowercase or uppercase. **
        ''')
        return sentence
    sentence = sentence.split()
    edited_sentence = []
    for word in sentence:
        if word in contractions:
            word = contractions[word]
        edited_sentence.append(word)
    return ' '.join(edited_sentence)

def lemmatize_sentence(sentence):
    import nltk
    from nltk.stem import WordNetLemmatizer
    wnl = WordNetLemmatizer()
    sentence = sentence.split()
    edited_sentence = []
    for word in sentence:
        edited_sentence.append(wnl.lemmatize(word, pos='v'))
    return " ".join(edited_sentence)

def predictionFor(text, model, tokenizer, max_length=128):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        truncation=True
    )

    input_ids = tf.constant([inputs['input_ids']], dtype=tf.int32)
    attention_mask = tf.constant([inputs['attention_mask']], dtype=tf.int32)
    token_type_ids = tf.constant([inputs['token_type_ids']], dtype=tf.int32)
    
    # Remove the [0] indexing when accessing the model's output
    logits = model({'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids})['logits']
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
    
    return probabilities

def resultsPrinter(
        text, 
        model,
        tokenizer,
        style=1,
        label_cat_dict={
            0: 'Crime',
            1: 'Neutral',
            2: 'Offensive',
            3: 'Premeditation',
            4: 'Safe',
            5: 'Unethical'
        }):
    probabilities = predictionFor(text, model, tokenizer)
    result = text + ' [' + label_cat_dict[np.argmax(probabilities)].upper() + ']'
    print(result)

def transform_text(text: str,
                  do_lower_case: bool=True,
                  do_remove_special_char_and_stopwords: bool=True,
                  do_expand_contractions: bool=True,
                  do_stemming: bool=False,
                  do_lemmatization: bool=False,
                  do_tokenize: bool=False):
    text = text.replace('-', ' ').replace('"', '').replace('.', ' ').replace('’', '\'').replace('”', '')
    
    if do_expand_contractions:
        text = expand_contractions(text)
    
    if do_lower_case:
        text = text.lower()
        
    if do_remove_special_char_and_stopwords:
        output = ''
        for i in text:
            if i.isalnum() or i == ' ':
                output += i
        text = output
        # text = nltk.word_tokenize(text)
        # output = []
        # for i in text:
        #     if i not in stopwords.words('english'):
        #         output.append(i)
        # text = ' '.join(output)
        
    if not (do_lemmatization and do_stemming):
        if do_lemmatization:
            text = lemmatize_sentence(text)
            
        if do_stemming:
            ps = PorterStemmer()
            words = nltk.word_tokenize(text)
            stemmed_words = [ps.stem(word) for word in words]
            text = ' '.join(stemmed_words)
            
    if do_tokenize:
        text = nltk.word_tokenize(text)
        
    return text

def create_tf_dataset(features, batch_size):
    input_ids = tf.constant([f.input_ids for f in features], dtype=tf.int32)
    attention_mask = tf.constant([f.attention_mask for f in features], dtype=tf.int32)
    token_type_ids = tf.constant([f.token_type_ids for f in features], dtype=tf.int32)
    labels = tf.constant([f.label for f in features], dtype=tf.int64)

    dataset = tf.data.Dataset.from_tensor_slices(({
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    }, labels))
    dataset = dataset.shuffle(buffer_size=len(features)).batch(batch_size)
    return dataset

def convert_examples_to_features(examples, tokenizer, max_length=128):
    features = []
    for example in examples:
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            truncation=True
        )

        features.append(
            InputFeatures(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids'],
                label=example.label
            )
        )

    return features

def create_input_example(text, label):
    return InputExample(guid=None,
                        text_a=text,
                        text_b=None,
                        label=label)

def clear_line():
    """Clear the current line in the terminal."""
    sys.stdout.write("\033[K")

def print_and_flush(string):
    """Print a string and flush the buffer immediately."""
    sys.stdout.write(string)
    sys.stdout.flush()

# Get the size of your terminal
columns, _ = shutil.get_terminal_size()

model = tf.keras.models.load_model('bert_model_v2')
tokenizer = BertTokenizer.from_pretrained('tokenizer')
# model = TFBertForSequenceClassification.from_pretrained('bert_model_v1', num_classes=6)

while True:
    choice = int(input('Enter 1 for train, 2 for test, 0 for exit: '))
    if choice == 1:
        file_name = "new_train_data.csv"
        
        new_conv = input("Enter the conversation: ")
        new_label = input("Enter the expected output (label): ")

        while True:
            are_you_sure = input("\nAre you sure? (y/n): ")

            if are_you_sure == 'y':
                new_df = pd.DataFrame({'conv': [new_conv], 'label': [new_label]})
                new_df.to_csv(file_name, header=True, index=False)
                new_train_data = pd.read_csv(file_name)
                new_train_data['transformed_conv'] = new_train_data['conv'].apply(transform_text, do_lemmatization=True)
                new_input_examples = [create_input_example(text, label) for text, label in zip(new_train_data['transformed_conv'], new_train_data['label'])]
                new_input_features = convert_examples_to_features(new_input_examples, tokenizer, max_length=128)
                train_dataset = create_tf_dataset(new_input_features, batch_size=1)
                optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
                model.fit(train_dataset, epochs=1)
                model.save('bert_model_v2')
                break
                print('\n')
            elif are_you_sure == 'n':
                break
                print('\n')
            else:
                print("** Enter the correct choice **")
        print('\n')
    elif choice == 2:
        text = input("Enter the conversation to predict: ")
        text = transform_text(text)
        print('\n')
        resultsPrinter(
            text = text,
            model = model,
            tokenizer = tokenizer,
            style = 0,
        )
        print('\n')
    else:
        print("\n GOOD BYE \n")
        break