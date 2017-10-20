import logging

import keras
from keras.engine import Model
from keras.layers import Dense, SimpleRNN, Embedding, Conv1D, MaxPooling1D, Flatten


def gen_rnn_model(articles, embedding_matrix, word_to_index):
    """
    Generate a recurrent neural network model, with an embedding layer.

    :param articles: A Pandas DataFrame containing the field padded_indices
    :type articles: pandas.DataFrame
    :param embedding_matrix: An embedding matrix, with shape (n,m), where n is the number of words, and m is the
    dimensionality of the embedding
    :type embedding_matrix: numpy.array
    :param word_to_index: A mapping from words (strings), to their index in the embedding matrix. For example
    embedding_matrix[word_to_index['pineapple']] would give the embedding vector for the word 'pineapple'
    :type: {str:int}
    :return: A keras model that can be trained on the given padded indices
    :rtype: Model
    """
    # Number of words in the word lookup index
    embedding_input_dim = embedding_matrix.shape[0]

    # Number of dimensions in the embedding
    embedding_output_dim = embedding_matrix.shape[1]

    # Maximum length of the x vectors
    embedding_input_length = max(articles['padded_indices'].apply(len))

    # Number of output labels
    output_shape = max(articles['category_encoded'].apply(len))

    logging.info('embedding_input_dim: {}, embedding_output_dim: {}, embedding_input_length: {}, '
                 'output_shape: {}'.format(embedding_input_dim, embedding_output_dim, embedding_input_length,
                                           output_shape))

    # Create architecture

    # Create embedding layer
    embedding_layer = Embedding(input_dim=embedding_input_dim,
                                output_dim=embedding_output_dim,
                                weights=[embedding_matrix],
                                input_length=embedding_input_length,
                                trainable=False)

    sequence_input = keras.Input(shape=(embedding_input_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = SimpleRNN(50)(embedded_sequences)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    preds = Dense(units=output_shape, activation='softmax')(x)

    # Compile architecture
    classification_model = Model(sequence_input, preds)
    classification_model.compile(loss='categorical_crossentropy',
                                 optimizer='rmsprop',
                                 metrics=['acc'])

    return classification_model


def gen_ff_model(articles, embedding_matrix, word_to_index):
    """
    Generate a feed forward neural network model, with an embedding layer.

    :param articles: A Pandas DataFrame containing the field padded_indices
    :type articles: pandas.DataFrame
    :param embedding_matrix: An embedding matrix, with shape (n,m), where n is the number of words, and m is the
    dimensionality of the embedding
    :type embedding_matrix: numpy.array
    :param word_to_index: A mapping from words (strings), to their index in the embedding matrix. For example
    embedding_matrix[word_to_index['pineapple']] would give the embedding vector for the word 'pineapple'
    :type: {str:int}
    :return: A keras model that can be trained on the given padded indices
    :rtype: Model
    """

    # Number of words in the word lookup index
    embedding_input_dim = embedding_matrix.shape[0]

    # Number of dimensions in the embedding
    embedding_output_dim = embedding_matrix.shape[1]

    # Maximum length of the x vectors
    embedding_input_length = max(articles['padded_indices'].apply(len))

    # Number of output labels
    output_shape = max(articles['category_encoded'].apply(len))

    logging.info('embedding_input_dim: {}, embedding_output_dim: {}, embedding_input_length: {}, '
                 'output_shape: {}'.format(embedding_input_dim, embedding_output_dim, embedding_input_length,
                                           output_shape))

    # Create architecture

    # Create embedding layer
    embedding_layer = Embedding(input_dim=embedding_input_dim,
                                output_dim=embedding_output_dim,
                                weights=[embedding_matrix],
                                input_length=embedding_input_length,
                                trainable=False)

    sequence_input = keras.Input(shape=(embedding_input_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Flatten()(embedded_sequences)
    x = Dense(512)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    preds = Dense(units=output_shape, activation='softmax')(x)

    # Compile architecture
    classification_model = Model(sequence_input, preds)
    classification_model.compile(loss='categorical_crossentropy',
                                 optimizer='rmsprop',
                                 metrics=['acc'])

    return classification_model


def gen_conv_model(articles, embedding_matrix, word_to_index):
    """
    Generate a convolutional neural network model, with an embedding layer.

    :param articles: A Pandas DataFrame containing the field padded_indices
    :type articles: pandas.DataFrame
    :param embedding_matrix: An embedding matrix, with shape (n,m), where n is the number of words, and m is the
    dimensionality of the embedding
    :type embedding_matrix: numpy.array
    :param word_to_index: A mapping from words (strings), to their index in the embedding matrix. For example
    embedding_matrix[word_to_index['pineapple']] would give the embedding vector for the word 'pineapple'
    :type: {str:int}
    :return: A keras model that can be trained on the given padded indices
    :rtype: Model
    """

    # Number of words in the word lookup index
    embedding_input_dim = embedding_matrix.shape[0]

    # Number of dimensions in the embedding
    embedding_output_dim = embedding_matrix.shape[1]

    # Maximum length of the x vectors
    embedding_input_length = max(articles['padded_indices'].apply(len))

    # Number of output labels
    output_shape = max(articles['category_encoded'].apply(len))

    logging.info('embedding_input_dim: {}, embedding_output_dim: {}, embedding_input_length: {}, '
                 'output_shape: {}'.format(embedding_input_dim, embedding_output_dim, embedding_input_length,
                                           output_shape))

    # Create architecture

    # Create embedding layer
    embedding_layer = Embedding(input_dim=embedding_input_dim,
                                output_dim=embedding_output_dim,
                                weights=[embedding_matrix],
                                input_length=embedding_input_length,
                                trainable=False)

    sequence_input = keras.Input(shape=(embedding_input_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    preds = Dense(units=output_shape, activation='softmax')(x)

    # Compile architecture
    classification_model = Model(sequence_input, preds)
    classification_model.compile(loss='categorical_crossentropy',
                                 optimizer='rmsprop',
                                 metrics=['acc'])

    return classification_model
