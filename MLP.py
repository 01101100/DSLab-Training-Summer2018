import tensorflow as tf

class MLP:
    def __init__(self, vocab_size, hidden_size):
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
    
    def build_graph(self):
        self._X = tf.placeholder(tf.float32, shape=[None, self._vocab_size])
        self._real_Y = tf.placeholder(tf.int32, shape=[None, ])

        weights_1 = tf.get_variable(
            name = 'weights_input_hidden',
            shape=(self._vocab_size, self._hidden_size),
            initializer=tf.random_normal_initializer(seed=2018),
        )

        biases_1 = tf.get_variable(
            name='biases_input_hidden',
            shape=(self._hidden_size),
            initializer=tf.random_normal_initializer(seed=2018)
        )

        weights_2 = tf.get_variable(
            name = 'weights_hidden_output',
            shape=(self._hidden_size, NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=2018),
        )

        biases_2 = tf.get_variable(
            name='biases_input_hidden',
            shape=(NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=2018)
        )

        hidden = tf.matmul(self._X, weights_1) + biases_1
        hidden = tf.sigmoid(hidden)
        logits = tf.matmul(hidden, weights_2) + biases_2

        labels_one_hot = tf.one_hot(indices=self._real_Y, depth=NUM_CLASSES, dtype=tf.float32)
        
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)

        loss = tf.reduce_mean(loss)

        probs = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(probs, axis=1)
        predicted_labels = tf.squeeze(predicted_labels)

        return predicted_labels, loss

    def trainer(self, loss, learning_rate):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_op

