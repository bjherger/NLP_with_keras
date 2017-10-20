#!/usr/bin/env python
"""
coding=utf-8

Main entry point into Moby-Dick project.

There aren't great batteries included examples for modeling text with deep learning, so I've built out this repo to
contain starter code for:

 - Text processing: Processing text to be utilized with keras (text pre-processing, converting to indices,
 padding) - Pre-trained embedding: Using a pre-trained text embedding (GoogleNews 300) with keras (translating words
 to a point in \mathbb{R}^{300}) - Convolutional architecture: Modeling text with a convolutional architecture (
 functionally similar to Ngrams) - RNN architecture: Modeling text with a Recurrent Neural Net (RNN) architecture (
 functionally similar to a rolling window)

"""
import glob
import logging
import ntpath
import os

from collections import defaultdict

import numpy
import pandas

from gensim.utils import simple_preprocess

import lib
import resources
import models


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.getLogger().setLevel(level=logging.DEBUG)

    # Extract data from upstream
    articles, embedding_matrix, word_to_index = extract()

    # Transform data for modeling
    articles, embedding_matrix, word_to_index = transform(articles, embedding_matrix, word_to_index)

    # Model data / predict on data
    model(articles, embedding_matrix, word_to_index)

    # TODO Load. What is using this down stream?
    load()
    pass


def extract():
    """
    Extract necessary data / resources from upstream. This method will:

     - Validate that newsgroup data set is available, and read in
     - Validate that text embeddings are available, and read in
     - Validate that text to embedding index lookup is available, and read in


    :return: observations, embedding_matrix, word_to_index
    :rtype: (pandas.DataFrame, numpy.array, dict)
    """

    logging.info('Begin extract')
    logging.info('Performing extract for batch: {}, from newgroup_path: {}'
                 .format(lib.get_batch_name(), lib.get_conf('newsgroup_path')))

    # Download resources

    # Confirm newsgroup data set is downloaded
    resources.download_newsgroup()

    # Confirm that embedding is downloaded
    resources.download_embedding()

    # Extract resources from file system

    # Newsgroup20: Get list of all candidate documents
    glob_pattern = os.path.join(lib.get_conf('newsgroup_path'), '*', '*')
    logging.info('Searching for glob_pattern: {}'.format(glob_pattern))
    document_candidates = glob.glob(glob_pattern)

    # Newsgroup20: Create observations data set
    observations = pandas.DataFrame(document_candidates, columns=['document_path'])
    logging.info('Shape of observations data frame created from glob matches: {}'.format(observations.shape))

    # Newsgroup20: Re-order rows
    observations = observations.sample(frac=1)

    # Newsgroup20: Subset number of observations, if it's a test run
    if lib.get_conf('test_run'):
        logging.info('Reducing file size for test run')
        observations = observations.head(100)
        logging.info('Test run number of records: {}'.format(len(observations.index)))

    # Embedding: Load embedding
    embedding_matrix, word_to_index = resources.create_embedding_matrix()
    logging.info('word_to_index max index: {}'.format(max(word_to_index.values())))

    # Archive schema and return
    lib.archive_dataset_schemas('extract', locals(), globals())
    logging.info('End extract')
    return observations, embedding_matrix, word_to_index


def transform(observations, embedding_matrix, word_to_index):
    """
    Transform data and resources to be ready for model consumption. This includes:

     - Pulling document category
     - Pulling raw text (stripped of headers)
     - Normalizing text, converting text to an array of tokens, and converting array of tokens to array of embedding
     indices
     - Padding array of indices, so that all arrays of indices are the same length

    :param observations: A Pandas DataFrame containing the field padded_indices
    :type observations: pandas.DataFrame
    :param embedding_matrix: An embedding matrix, with shape (n,m), where n is the number of words, and m is the
    dimensionality of the embedding
    :type embedding_matrix: numpy.array
    :param word_to_index: A mapping from words (strings), to their index in the embedding matrix. For example
    embedding_matrix[word_to_index['pineapple']] would give the embedding vector for the word 'pineapple'
    :type: {str:int}
    :return: articles, embedding_matrix, word_to_index
    :rtype: (pandas.DataFrame, numpy.array, dict)
    """

    logging.info('Begin transform')

    # Transform embedding resources
    # Embedding: Update embedding to map any unknown words (words not in training vocabulary) to the unknown value
    default_dict_instance = defaultdict(lambda: word_to_index['UNK'])
    default_dict_instance.update(word_to_index)
    word_to_index = default_dict_instance

    # Transform newsgroup20 data set
    # Newsgroup20: Extract article filename from document path
    observations['filename'] = observations['document_path'].apply(lambda x: ntpath.basename(x))

    # Newsgroup20: Extract article category from document path
    observations['category'] = observations['document_path'].apply(lambda x: ntpath.basename(os.path.dirname(x)))

    # Newsgroup20: Extract article text (and strip article headers), from document path
    observations['text'] = observations['document_path'].apply(lambda x: lib.strip_header(open(x).readlines()))

    # Newsgroup20: Convert text to normalized tokens. Unknown tokens will map to 'UNK'.
    observations['tokens'] = observations['text'].apply(simple_preprocess)

    # Newsgroup20: Convert tokens to indices
    observations['indices'] = observations['tokens'].apply(lambda token_list: map(lambda token: word_to_index[token],
                                                                                  token_list))
    observations['indices'] = observations['indices'].apply(lambda x: numpy.array(x))

    # Newsgroup20: Pad indices list with zeros, so that every article's list of indices is the same length
    observations['padded_indices'] = observations['indices'].apply(lib.pad_sequence)

    # Newsgroup20: Create label one hot encoder, and encode labels
    label_encoder = lib.create_label_encoder(set(observations['category']))
    observations['category_encoded'] = observations['category'].apply(lambda x: label_encoder[x])

    # Set up modeling input
    observations['modeling_input'] = observations['padded_indices']

    # Archive schema and return
    lib.archive_dataset_schemas('transform', locals(), globals())
    logging.info('End transform')
    return observations, embedding_matrix, word_to_index


def model(observations, embedding_matrix, word_to_index):
    """
    Train models, as appropriate.

    The Pandas DataFrame observations should contain the fields `modeling_input` (where each member is an iterable of
    the same length), and `category_encoded` (where each member is an iterable of the same length)

    :param observations: A Pandas DataFrame containing the field padded_indices
    :type observations: pandas.DataFrame
    :param embedding_matrix: An embedding matrix, with shape (n,m), where n is the number of words, and m is the
    dimensionality of the embedding
    :type embedding_matrix: numpy.array
    :param word_to_index: A mapping from words (strings), to their index in the embedding matrix. For example
    embedding_matrix[word_to_index['pineapple']] would give the embedding vector for the word 'pineapple'
    :type: {str:int}
    :return: None
    :rtype: None
    """

    logging.info('Begin model')

    # Reference variables

    # Create train and test data sets

    train_test_mask = numpy.random.random(size=len(observations.index))
    num_train = sum(train_test_mask < .8)
    num_validate = sum(train_test_mask >= .8)
    logging.info('Proceeding w/ {} train observations, and {} test observations'.format(num_train, num_validate))

    x_train = observations['modeling_input'][train_test_mask < .8].tolist()
    y_train = observations['category_encoded'][train_test_mask < .8].tolist()
    x_test = observations['modeling_input'][train_test_mask >= .8].tolist()
    y_test = observations['category_encoded'][train_test_mask >= .8].tolist()

    # Convert x and y vectors to numpy objects
    x_train = numpy.array(x_train, dtype=object)
    y_train = numpy.array(y_train)
    x_test = numpy.array(x_test, dtype=object)
    y_test = numpy.array(y_test)

    logging.info('x_train shape: {}, y_train shape: {}, '
                 'x_test shape: {}, y_test shape: {}'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    # If required, train model
    if lib.get_conf('train_model'):
        logging.info('Creating and training model')

        classification_model = models.gen_conv_model(observations, embedding_matrix, word_to_index)

        # Train model
        classification_model.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_test, y_test))

        logging.info('Finished creating and training model')

    # Validate model

    # Archive schema and return
    lib.archive_dataset_schemas('transform', locals(), globals())
    logging.info('End model')
    pass


def load():
    pass


# Main section
if __name__ == '__main__':
    main()
