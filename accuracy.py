import tensorflow as tf
import helpers
from DataReader import DataReader

if __name__ == "__main__":
    test_data_reader = DataReader(
        data_path="/datasets/20news-test-tfidf.txt",
        batch_size=50,
        vocab_size=10
    )

    with tf.Session() as sess:
        epoch = 10

        trainable_variables = tf.trainable_variables()
        for variable in trainable_variables:
            saved_value = helpers.restore_parameters(variable.name, epoch)
            assign_op = variable.assign(saved_value)
            sess.run(assign_op)