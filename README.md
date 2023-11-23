# NLP AI

This git repository is part of a project that implements a chatbot which
impersonates a patient who suffers from cardiovascular diseases.

The project consists of three repositories which contain the logic of the
application:

- Client [nlp-ui](https://github.com/eduard-gp/nlp-ui)
- Server [nlp-server](https://github.com/eduard-gp/nlp-server)
- AI (**this repository**)

## Application Arhitecture

The application has an arhitecture of client-server. There are three components:
a user interface, a web server and a database. The user interacts with the
chatbot using the UI. It is possible to perform the following actions:

- Login
- Create a new patient
- Modify the information of a patient that already exists
- Interact with a patient

The following image describes the workflow of the application:

![Application workflow](images/workflow.jpg)

## NLP AI

This repository contains the details of how to train deep neural networks and
the results that were obtained. The goal of project is to obtain a chatbot that
can impersonate a patient and answer questions from a medical student. The
result of training a deep neural network is a model. The models in this
repository are trained for text classification and named entity recognition. The
task of text classification is used to identify the intention of a question. The
following intentions are of interest for our project:

- chest_pain
- cough_symptoms
- diseases_parents
- diseases_treatment
- faintint_symptoms
- fever_symptoms
- greetings
- palpitations_symptoms
- surgeries
- symptoms_changes
- symptoms_circumstances
- symptoms_parents
- symptoms_start
- visit_reason

For the task of named entity recognition for the Romanian language the labels
from *Dumitrescu, S. D.-M. (2019). Introducing RONEC--the Romanian Named Entity
Corpus. arXiv preprint arXiv:1909.01247* are used.

- PERSON
- NAT_REL_POL
- ORGANIZATION
- GPE
- LOC
- FACILITY
- PRODUCT
- EVENT
- LANGUAGE
- WORK_OF_ART
- DATETIME
- PERIOD
- MONEY
- QUANTITY
- NUMERIC_VALUE
- ORDINAL

For the task of named entity recognition for the English language the labels
from *Erik F. Tjong Kim Sang, F. D. (2003). Introduction to the CoNLL-2003
Shared Task: Language-Independent Named Entity Recognition. arXiv:cs/0306050*
are used.

- LOC
- MISC
- ORG
- PER

## Neural networks

There are two types of neural networks that are trained for the two task
mentioned aboved:

- Recurrent Neural Networks
- Transformers (BERT)

Recurrent neural networks (RNN) are a special type of neural networks that are
specialised to process sequential data and can process data of variable length.
A recurent neural network can be described as a computational graph.

![Computational graph of a RNN](images/rnn_graph.jpg)

x^(t)^ is the input to time t, h^(t)^ is the hidden state of the RNN at time t
and o^(t)^ is the output at time t. The hidden state of the RNN has the task of
memory so it can retain the information learnt in the previous steps.

Transformers are neuronal networks tha use the concept of attention. The natural
language is complex and some words have a different meaning depending on context
even thought they have the same form. For example, "I go to the bank." and "I
can see the river bank.", the word "bank" has a different meaning in the two
sentences, but has the same form. The concept of attention takes into account
the context where a word is. For moree information please read *Ashish Vaswani,
N. S. (2017). Attention Is All You Need. arXiv:1706.03762*.

BERT or Bidirectional Encoder Representation is a type of transformer that uses
the concept of attention. BERT is a big model that has millions of parameteres.
The model also masks some words when it is the training phase. For further
details about how the models is implemented. please read the original paper
*Jacob Devlin, M.-W. C. (2018). BERT: Pre-training of Deep Bidirectional
Transformers for Language Understanding. arXiv:1810.04805*.

### Datasets

The datasets for training are in the directories:

- datasets (used for named entity recognition)
- database/text_classification (used for text_classification)

The datasets in database/text_classification were processed from raw data from
wit_ai. The directory database/dialogs show examples of how a dialog between a
student and patient should be. The scripts for processing the raw data are
located in the database directory. Also, the special syntax used the dialogs is
exmplained in the README.md from that directory.

The data exploratory analyzes for the task of text classification is presented below.

| Dataset text classification | Items | Average character length per item | Unique words |
|-----------------------------|-------|-----------------------------------|--------------|
| Romanian                    | 215   | 31.44                             | 299          |
| English                     | 95    | 31.35                             | 147          |

The following diagrams describe the dataset for the Romanian language.

![Count of labels per label category Romanian](images/text_classification_labels_per_category_ro.jpg)

![Mean length of sentence per label category Romanian](iamges/text_classification_mean_length_ro.jpg)

The following diagrams describe the dataset for the English language.

![Count of labels per label category English](images/text_classification_labels_per_category_en.jpg)

![Mean lenght of sentence per label category English](images/text_classification_mean_length_en.jpg)

The data exploratory analyzes for the task of named entity recognition is presented below.

