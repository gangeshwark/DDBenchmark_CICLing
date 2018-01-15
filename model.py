import tensorflow as tf
import numpy as np

import load_data

np.random.seed(100000)
# i = int(121 * 0.8)
# print(i)
# i = 100
X_train_text, X_test_text, X_train_audio, X_test_audio, X_train_gest, X_test_gest, X_train_video, X_test_video, Y_train, Y_test = load_data.load()

print(X_train_audio.shape, X_test_audio.shape)

batch_size = 50
text_input_dim = 300
audio_input_dim = 300
video_input_dim = 300
gestures_input_dim = 39
classes = 2
hidden_size = (1024, 512)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)


fusion = 'd'  # c for concat, d for dot


def create_model():
    # create placeholders
    X_audio = tf.placeholder(tf.float32, shape=(None, audio_input_dim), name='audio_input')
    X_text = tf.placeholder(tf.float32, shape=(None, text_input_dim), name='text_input')
    X_video = tf.placeholder(tf.float32, shape=(None, text_input_dim), name='video_input')
    X_gest = tf.placeholder(tf.float32, shape=(None, gestures_input_dim), name='gest_input')
    y = tf.placeholder(tf.int32, shape=(None, classes), name='targets')

    # X_audio1 = tf.nn.sigmoid(tf.layers.dense(X_audio, 256))
    # X_text1 = tf.nn.sigmoid(tf.layers.dense(X_text, 256))
    # X_video1 = tf.nn.sigmoid(tf.layers.dense(X_video, 256))
    # X_gest1 = tf.nn.sigmoid(tf.layers.dense(X_gest, 256))

    # with tf.variable_scope('Input') as scope:
    if fusion == 'c':
        X = tf.concat([X_audio, X_text, X_video, X_gest], axis=1, name='Input')
    elif fusion == 'd':
        # X = tf.concat([tf.multiply(X_audio1, tf.multiply(X_text1, X_video1)), X_gest1], axis=1, name='Input')
        X = tf.concat([tf.multiply(X_audio, tf.multiply(X_text, X_video)), X_gest], axis=1)  # , axis=1, name='Input')

    h1 = tf.nn.relu(tf.layers.dense(X, hidden_size[0]))
    h1 = tf.layers.dropout(h1, 0.5)

    # h1 = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    # h2 = tf.matmul(h1, w_2)
    # h1_ = tf.nn.dropout(h1, 0.5)

    # h2 = tf.nn.tanh(tf.matmul(h1_, w_2))  # The \sigma function
    # h2_ = tf.nn.dropout(h2, 0.5)
    # yhat = tf.matmul(h2_, w_3)
    # for i, h in enumerate(hidden_size):

    # h2 = tf.nn.relu(tf.layers.dense(h1, hidden_size[1]))
    # h2 = tf.layers.dropout(h2, 0.5)
    # h3 = tf.nn.relu(tf.layers.dense(h2, hidden_size[2]))
    # h3 = tf.layers.dropout(h3, 0.5)
    # h4 = tf.nn.relu(tf.layers.dense(h3, hidden_size[3]))
    # h4 = tf.layers.dropout(h4, 0.5)

    # h5 = tf.nn.relu(tf.layers.dense(h4, hidden_size[3]))
    # h5 = tf.layers.dropout(h5, 0.5)
    logits = tf.layers.dense(h1, classes)
    # softmaxed_logits = tf.nn.softmax(logits)
    predict = tf.argmax(logits, axis=1)

    auc = tf.metrics.auc(labels=tf.argmax(y, 1), predictions=predict)
    class_accuracy = tf.metrics.mean_per_class_accuracy(labels=tf.argmax(y, 1), predictions=predict,
                                                        num_classes=2)

    # Backward propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    # Run SGD
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    max_test_acc = 0.0
    max_test_epoch = 100
    print("Training...")

    for epoch in range(2000):
        # Train with each example
        losses = []
        for i in range(0, X_train_audio.shape[0], batch_size):
            _, loss = sess.run([updates, cost],
                               feed_dict={X_audio: X_train_audio[i: i + batch_size],
                                          X_text: X_train_text[i:i + batch_size],
                                          X_video: X_train_video[i:i + batch_size],
                                          X_gest: X_train_gest[i:i + batch_size],
                                          y: Y_train[i: i + batch_size]})
            losses.append(loss)
        train_accuracy = np.mean(
            np.argmax(Y_train, axis=1) == sess.run(predict, feed_dict={X_audio: X_train_audio,
                                                                       X_text: X_train_text,
                                                                       X_video: X_train_video,
                                                                       X_gest: X_train_gest,
                                                                       y: Y_train}))
        test_accuracy = np.mean(np.argmax(Y_test, axis=1) == sess.run(predict,
                                                                      feed_dict={X_audio: X_test_audio,
                                                                                 X_text: X_test_text,
                                                                                 X_video: X_test_video,
                                                                                 X_gest: X_test_gest,
                                                                                 y: Y_test}))

        auc_score = sess.run(auc, feed_dict={X_audio: X_test_audio,
                                             X_text: X_test_text,
                                             X_video: X_test_video,
                                             X_gest: X_test_gest,
                                             y: Y_test})

        if test_accuracy > max_test_acc and max_test_epoch < epoch:
            max_test_acc = test_accuracy
            max_test_epoch = epoch

        print("Epoch = %d, loss = %.5f train accuracy = %.2f%%, test accuracy = %.2f%%, AUC = %.5f"
              % (epoch + 1, np.mean(losses), 100. * train_accuracy, 100. * test_accuracy, np.mean(auc_score)))
        print(auc_score)
    out = sess.run(predict,
                   feed_dict={X_audio: X_test_audio,
                              X_text: X_test_text,
                              X_video: X_test_video,
                              X_gest: X_test_gest,
                              y: Y_test})
    class_acc = sess.run(class_accuracy,
                         feed_dict={X_audio: X_test_audio,
                                    X_text: X_test_text,
                                    X_video: X_test_video,
                                    X_gest: X_test_gest,
                                    y: Y_test})
    actual = np.argmax(Y_test, axis=1)
    print(out)
    print(actual)
    confusion = tf.confusion_matrix(actual, out, num_classes=2)
    # actual = tf.convert_to_tensor(actual, tf.int64)
    # out = tf.convert_to_tensor(out, tf.int64)
    # sess.run(tf.initialize_all_variables())
    # class_accuracy = tf.metrics.mean_per_class_accuracy(actual, out,
    #                                                     num_classes=2)
    mat = sess.run(confusion)
    print(mat)
    print(mat.diagonal() / mat.sum(1))
    # print(class_acc)
    # auc = tf.metrics.auc(actual, out)
    # print(sess.run(auc))

    """
    test_feed_dict = {X_audio: X_audio_test,
                      X_text: X_text_test,
                      y: Y_test}
    test_pred = sess.run(predict, feed_dict=test_feed_dict)
    test_accuracy = np.mean(np.argmax(Y_test, axis=1) == test_pred)
    print "test_accuracy", test_accuracy
    """
    print("Max Test Accuracy: ", max_test_acc, "at epoch", max_test_epoch)
    sess.close()


if __name__ == '__main__':
    create_model()
