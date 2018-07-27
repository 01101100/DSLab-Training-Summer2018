from MLP import MLP
from DataReader import DataReader
import tensorflow as tf
import helpers
import numpy as np

def load_dataset():
    train_data_reader = DataReader(
        data_path="/datasets/20news-train-tfidf.txt",
        batch_size=50,
        vocab_size=vocab_size
    )

    test_data_reader = DataReader(
        data_path="/datasets/20news-test-tfidf.txt",
        batch_size=50,
        vocab_size=vocab_size
    )

    return train_data_reader, test_data_reader


if __name__ == "__main__":
    with open("/datasets/words_idfs.txt") as f:
        vocab_size = len(f.read().splitlines())

    mlp = MLP(
        vocab_size=vocab_size,
        hidden_size=50
    )

    predicted_labels, loss = mlp.build_graph()
    train_op = mlp.trainer(loss=loss, learning_rate=0.01) # best learning rate is 10e-4

    with tf.Session() as sess:
        train_data_reader, test_data_reader = load_dataset()
        step, MAX_STEP = 0, 1000 ** 2

        sess.run(tf.global_variables_initializer())
        while step < MAX_STEP:
            train_data, train_labels = train_data_reader.next_batch()
            plavels_eval, loss_eval, _ = sess.run(
                [predicted_labels, loss, train_op],
                feed_dict={
                    mlp._X: train_data,
                    mlp._real_Y: train_labels
                }
            )
            step += 1
            print("step: {}, loss: {}".format(step, loss_eval))

        trainable_variables = tf.trainable_variables()
        for variable in trainable_variables:
            helpers.save_parameters(
                name=variable.name,
                value=variable.eval(),
                epoch=train_data_reader._num_epoch
            )

    with tf.Session() as sess:
        epoch = 10

        trainable_variables = tf.trainable_variables()
        for variable in trainable_variables:
            saved_value = helpers.restore_parameters(variable.name, epoch)
            assign_op = variable.assign(saved_value)
            sess.run(assign_op)

        num_true_preds = 0
        while True:
            test_data, test_labels = test_data_reader.next_batch()
            test_plabels_eval = sess.run(
                predicted_labels,
                feed_dict={
                    mlp._X: test_data,
                    mlp._real_Y: test_labels
                }
            )
            matches = np.equal(test_plabels_eval, test_labels)
            num_true_preds += np.sum(matches.astype(float))

            if test_data_reader._batch_id == 0:
                break

            print("Epoch:", epoch)
            print("Accuracy on test data: ", num_true_preds / len(test_data_reader._data))