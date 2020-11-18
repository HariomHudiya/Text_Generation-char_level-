import tensorflow as tf
import numpy as np
import os

print("-----------------------------------------------------")
file_path = "./datasets/shakespeare.txt"
def file_reader(file_path):
    data = open(file_path, 'rb').read().decode(encoding='utf-8')
    print("File accessed Successfully.")
    return data

data = file_reader(file_path)

# path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# data = open(path_to_file,'rb').read().decode(encoding='utf-8')
print('Length of text: ' , len(data))

# first 100 characters in data
print("First 100 character :")
print(data[:100])

print("-----------------------------------------------------")

def character_vocab(data):
    """
    Build character level vocab
    :param data: Raw data
    :return: vocab
    """""
    print("Creating Character Based Vocab")
    vocab = sorted(set(data))
    print('{} unique characters'.format(len(vocab)))
    print("unique character")
    print(vocab)
    return vocab

vocab = character_vocab(data)

# Since Unique characters are beyond normal ABC,abc and 1-9 we have first opened file in binary and then decoded in UTF-8
# to incoporate wider range of characters
print("-----------------------------------------------------")

# Creating mappings

def mappings(vocab):
    """

    :param vocab: Vocab
    :return: char_to_idx(dict) , idx_to_char(dict) and idx2char(list)
    """
    print("Building Mappings....")
    char_to_idx = {char : idx for idx,char in enumerate(vocab)}
    idx_to_char = {idx : char for idx,char in enumerate(vocab)}
    idx2char = np.array(vocab)
    print("length of vocab: ", len(char_to_idx)) # for debugging
    print(char_to_idx)
    return char_to_idx,idx_to_char,idx2char

char_to_idx,idx_to_char,idx2char = mappings(vocab)

print("-----------------------------------------------------")

def data_2_int(char_to_idx,data):
    """

    :param char_to_idx: dict to convert character to index
    :return: Entire Data to int
    """
    print("Converting Whole data as int..")
    # Converting whole text data to integer data (at char level)
    data_as_int = np.array([char_to_idx[char] for char in data])
    return data_as_int

data_as_int = data_2_int(char_to_idx,data)

print("First 16 chars --",data[:16])
print("Mapping of first 16 chars--",data_as_int[:16])

# Debugging info

# print("------------------Char_to_index-------------------------")
# print('{')
# for char,_ in zip(char_to_idx, range(20)):
#     print('  {:4s}: {:3d},'.format(repr(char), char_to_idx[char]))
# print('  ...\n}')
#
# print("------------------index_to_char-------------------------")
#
# for idx in range(20):
#     print(idx,str(idx_to_char[idx]))

sentence = "First Citizen"
print("Sentence -- ", sentence)
print("Mapping--",[char_to_idx[char] for char in sentence])

# Task
# Given a character or Sequnece of characters, what is the most probable next character ?

# Creating Training and targets example

def sequnce_dataset(data_as_int,seq_length = 100):
    """

    :param data_as_int: integer representation of data
    :param seq_length: Input and Target length for Training_set
    :return:
    """
    print("Building Sequnce data object")
    # The maximum number of characters we want for a single input
    seq_length = seq_length
    # converting numpy integer_data to tensorflow obj
    char_dataset = tf.data.Dataset.from_tensor_slices(data_as_int)
    print("1 ----- Building Character Dataset in Tensorflow")



    # Debugging dataset obj
    for i in char_dataset.take(5):
        print(idx_to_char[i.numpy()])

    # Making Batched dataset

    sequences = char_dataset.batch(seq_length + 1,drop_remainder=True)  # Sequence length +1 ,coz for Target we need right shift so that both input and output are of 100

    counter = 1

    for item in sequences.take(5):
        # print(repr(''.join(idx_to_char[item.numpy()])))
        item_len = len(item)
        print("Batch number", counter)
        print("Batch char_length : ", item_len)

        # print(idx_to_char([idx for idx in item.numpy()]))
        print(repr(''.join(idx2char[item.numpy()])))

        print("------------------")
        counter = counter + 1

    return sequences


sequences = sequnce_dataset(data_as_int,100)


# Helper Function
def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text,target_text

# Mapping function to "Bathced" sequences

dataset = sequences.map(split_input_target)
# Let's check the working of the above function

for input_example,target_example in dataset.take(1):
    print("---------------Input Text:-----------")
    print(len(input_example))
    print(''.join(idx2char[input_example.numpy()]))
    print("--------------Target Text:-------------------")
    print(len(target_example))
    print(''.join(idx2char[target_example.numpy()]))

#input_example[:5],target_example[:5] in dataset[:5]

# input_example[:5],target_example[:5] = dataset.take(5)
# dataset --- > batch, len(seq)       , input_example -- > len(seq) , input_eample[:5] -- > first 5 chars in input_example

for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))


# Upto Now we have batched chars to make batched_sequence
# Now We'll batch sequence to make training_bathces
# We will be processing sequnces not characters

batch_size = 64          # Take 64 Sequences , each of length 101 (seq_length + 1 )
buffer_size = 1000 # from 1000 shuffled training_sets pick 1 bathc(64 sequnces)

dataset = dataset.shuffle(buffer_size).batch(batch_size,drop_remainder=True)

# Exhaustive Dataset Object
# Once the data is over it exhausts
print("-------------------")
print(dataset)

# Building the model
print("------------Building Model-----------")

vocab_size = len(vocab) # unique characters

# the embedding dim

embedding_dim = 256 # number of units to represent a char

# Number of RNN units

rnn_units = 1024  # Number of units used to represent the hidden state of Rnn_cell

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=batch_size)

print("---------------Model Summary-----------------")
print(model.summary())
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print("Output_Prediction shape:",example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

"# (batch_size, sequence_length, vocab_size) --> for 1 batch consider 64 sequnces"
"and for each sequnce consider 100 charcters and each character is represented by an int(total 65 unique chars)"

# Let's see how our model performs in 1st input_example_batch
# We have corresponding output (of 1_st_input_example) example_batch_predictions

sampled_indices = tf.random.categorical(example_batch_predictions[0],num_samples=1)

# Squeezing to make 2d --> 1d and convertin to numpy

sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
print(sampled_indices)

print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))

####### Training the Model ###########

def loss(labels,predictions):
    return tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                           predictions,
                                                           from_logits=True)

# Calling loss to check execution

example_batch_loss = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())


model.compile(optimizer='adam', loss=loss)

# Configuring Checkpoints
# Directory where the checkpoints will be saved
checkpoint_dir = './shakespeare_chekpoint'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

epochs = 10
print("--------------Training Starting------------")

#history = model.fit(dataset,epochs=epochs,callbacks=[checkpoint_callback])

#print(tf.train.latest_checkpoint(checkpoint_dir))


model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

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