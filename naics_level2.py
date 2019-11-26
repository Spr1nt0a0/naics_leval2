#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import learn

# Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./model_saved/checkpoints/", "Checkpoint directory path")
FLAGS = tf.flags.FLAGS

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

# Put your own data here
x_raw = ["a masterpiece four years in the making", "everything is off."]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

# Prediction
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        batches = batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

print(all_predictions)

# 0 Agriculture, Forestry, Fishing and HuntingT
# 1 Mining, Quarrying, and Oil and Gas ExtractionT
# 2 UtilitiesT
# 3 ConstructionT
# 4 Wholesale TradeT
# 5 InformationT
# 6 Finance and InsuranceT
# 7 Real Estate and Rental and LeasingT
# 8 Professional, Scientific, and Technical ServicesT
# 9 Management of Companies and EnterprisesT
# 10 Administrative and Support and Waste Management and Remediation ServicesT
# 11 Educational ServicesT
# 12 Health Care and Social AssistanceT
# 13 Arts, Entertainment, and RecreationT
# 14 Accommodation and Food ServicesT
# 15 Other Services (except Public Administration)T
# 16 Public AdministrationT
