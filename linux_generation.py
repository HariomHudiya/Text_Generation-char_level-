import tensorflow as tf
import numpy as np
import os

def file_reader(file_path,debugging_info = True):
    """

    :param file_path: file path
    :return: data
    """
    data = open(file_path).read()
    print("Data Accessing Successful")

    if debugging_info:
        print("Length of Data: ", len(data))
    return data

file_path = "./datasets/linux.txt"
data = file_reader(file_path)

print("---------------------------------------")

def vocab_builder(data,debugging_info =True):
    print("Building Character based vocab: ")
    vocab = sorted(set(data))               # Sorted so they occur in specific order and not in order of encounter
    if debugging_info:
        print("Vocab Length :",len(vocab))
        print("Unique Characters in dataset ")
        print(vocab)
    return vocab

vocab = vocab_builder(data)
print("------------------------------------------")

def mappings(vocab,debugging_info = True):
    """

    :param vocab: Vocabulary
    :param debugging_info:
    :return: char_to_idx(dict) , idx_to_char(dict) and idx2char (list)
    """
    print("Building Mappings.....")

    char_to_idx = {char : idx for idx,char in enumerate(vocab)}
    idx_to_char = {idx : char for idx,char in enumerate(vocab)}

    idx2char = np.array(vocab)

    if debugging_info:
        print("Length of char_to_idx Mapping ",) # 97
        print("char_to_idx ",len(char_to_idx) )
        print(char_to_idx)
        print("idx_to_char ",len(idx_to_char))
        print(idx_to_char)
        print("idx2char ", len(idx2char))
        print(idx2char)

    return char_to_idx,idx_to_char,idx2char

print("--------------------------------------------------")

char_to_idx, idx_to_char, idx2char = mappings(vocab,debugging_info=False)

def data2int(char_to_idx, data,debugging_info= True):
    """

    :param char_to_idx:
    :param data:
    :return: integer representation of data
    """
    print("Converting data to integer representation ")
    data_int = [char_to_idx[char] for char in data]

    if debugging_info:
        print("Data_int")
        print("Length of data_int",len(data_int))  #999588 (len of data)
        #print(data_int)
    return data_int

print("----------------------------------------------")
data_int = data2int(char_to_idx,data)

def char_dataset_builder(data_int,idx_to_char,debugging_info=True):
    """

    :param data_int: Integer representation of data
    :param char_to_idx: Used here for debugging purposes
    :param debugging_info:
    :return:
    """

    char_dataset = tf.data.Dataset.from_tensor_slices(data_int)

    if debugging_info:
        print("Character Dataset Object") # it is filled with scalers hence shape = 0
        print(char_dataset)

        print("First five Characters in the char_dataset")
        # Since i is tensor (dataset obj) it is unhashable so we convert i to numpy(tensor obj property)
        [print(idx_to_char[i.numpy()]) for i in char_dataset.take(5)]

    return char_dataset

char_dataset = char_dataset_builder(data_int,idx_to_char,debugging_info=False)

print("----------------------------------------------------------")

def seq_builder(char_dataset, seq_length = 100,debugging_info =True ):

    # This is the length that will be used for inputs and targets in the training set

    print("Building Sequnce Dataset .....")
    seq_length = seq_length
    sequences_dataset = char_dataset.batch(seq_length + 1 , drop_remainder=True)
    if debugging_info:
        print("Sequnces Dataset")
        print(sequences_dataset)

        print("First 5 items in Sequence Dataset.")
        counter = 1
        for item in sequences_dataset.take(5):

            print("Item Number ", counter , "Length of Item ",len(item))
            print(item)
            counter += 1

    return sequences_dataset

sequence_dataset = seq_builder(char_dataset)

print("----------------------------------------------------------------")
def input_target_splitter(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text,target_text

def dataset_splitter(sequence_dataset, input_target_splitter,idx_to_char,batch_size, debugging_info = True):
    """

    :param sequence_dataset: Sequence dataset object
    :param input_target_splitter: helper function to split input and targets
    :return: input_target pair dataset object in batches
    """
    print("Dataset Built as input and target pair....")
    dataset = sequence_dataset.map(input_target_splitter)

    buffer_size = 1000  # from 1000 shuffled training_sets(sequences) pick 1 batch(64 sequnces)

    dataset = dataset.shuffle(buffer_size).batch(batch_size=batch_size, drop_remainder=True)
    if debugging_info:
        for input_text,target_text in dataset.take(1):  # Consider 1st batch in dataset
            print("Input text :")    # len ---> 100
            print(input_text.numpy()[0])                # Taking 1st elment in 1st batch
            print("".join([idx_to_char[idx] for idx in input_text.numpy()[0]]))
            print("Target text: ")      # len ---> 100
            print(target_text.numpy()[0])
            print(''.join([idx_to_char[idx] for idx in target_text.numpy()[0]]))
            print("Despite shuffling sequnces are intact check completed")

    return dataset

dataset = dataset_splitter(sequence_dataset,input_target_splitter, idx_to_char,batch_size=64,debugging_info=False)


print("---------------------------------------------")

def model_builder(vocab_size,embedding_dim,rnn_units,batch_size):
    """

    :param vocab_size: Pass in vocab length
    :param embedding_dim: Number of units used to represent a character
    :param rnn_units: Number of units used to represent hidden state
    :param batch_size:
    :return:
    """
    print("Building Model...........")
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim,
                                  batch_input_shape=[batch_size,None]
                                  ),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = model_builder(vocab_size=len(vocab),
                      embedding_dim=256,
                      rnn_units=512,
                      batch_size=64)

# print("Evaluating Untrained Model --------")
#
# for input_text,target_text in dataset.take(1):
#     predicted_text = model(input_text)
#     print("Input text: ")
#     print("".join([idx_to_char[idx] for idx in input_text.numpy()[0]]))
#     print("Target text: ")
#     print("".join([idx_to_char[idx] for idx in target_text.numpy()[0]]))
#     print("Predicted Text: ")
#     print("".join([idx_to_char[idx] for idx in predicted_text.numpy()[0]]))


def loss(labels,predictions):
    return tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                           predictions,
                                                           from_logits=True)


model.compile(optimizer='adam', loss=loss)

print("---------------Training Model --------------")
# Configuring Checkpoints
# Directory where the checkpoints will be saved
checkpoint_dir = './linux_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

epochs = 10

#history = model.fit(dataset,epochs=epochs,callbacks=[checkpoint_callback])

print(tf.train.latest_checkpoint(checkpoint_dir))


model = model_builder(vocab_size=len(vocab),
                      embedding_dim=256,
                      rnn_units=512,
                      batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

print(model.summary())

def generate_text(model, start_string,num_generate = 500):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = num_generate

    # Converting our start string to numbers (vectorizing)
    input_eval = [char_to_idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    print("Input Dimension: ", input_eval.shape)

    # Empty string to store our results
    text_generated = []

    # Low temperature results in more predictable text.
    # Higher temperature results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        # The Input text can be of any variable length
        # It used for next character prediction.

        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        # Draws sample from categorical distribution
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # Pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


print(generate_text(model, start_string=u"Harry it is working,great !!"))

print("--------------------------------------------------------------------------------------")
print("This model is prone to characters not in vocab ")
print("Try any integer in the starting string....to verify Key:Error")
print("Also it ends abruptly without sentence completion..")
