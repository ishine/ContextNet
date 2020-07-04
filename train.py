import argparse
import rnnt_loss
import tensorflow as tf

from dataset import create_dataset
from model import ConvBlock, ContextNet

def create_conv_blocks():
    # TODO Audio encoder construction logic
    return []

def create_model(**kwargs):
    kwargs["create_conv_blocks"] = create_conv_blocks
    return ContextNet(kwargs)

def create_optimizer(lr):
    # TODO Transformer learning rate schedule
    return tf.keras.optimizers.Adam(learning_rate=lr)

def train(num_units, num_vocab, num_lstms, lstm_units, out_dim,
          lr, batch_size, num_epochs, data_path, vocab, mean, std_dev):
    model = create_model(num_units, num_vocab, create_conv_blocks,
                         num_lstms, lstm_units, out_dim)

    dev_dataset = create_dataset(data_path, "dev", vocab, mean, std_dev, batch_size)
    train_dataset = create_dataset(data_path, "train", vocab, mean, std_dev, batch_size)

    optimizer = create_optimizer(lr)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, './ckpt', max_to_keep=10)

    def dev_step(x, y, x_len, y_len):
        logits = model(x, y, x_len, y_len, training=False)
        # TODO Pass correct arguments
        # TODO Check if softmax or logit needs to be passed
        # TODO Implement greedy decoding for error
        loss = rnnt_loss(logits, y)
        error = 0
        return loss, error

    def train_step(x, y, x_len, y_len):
        with tf.GradientTape() as tape:
            logits = model(x, y, x_len, y_len, training=True)
            # TODO Pass correct arguments
            # TODO Check if softmax or logit needs to be passed
            # TODO Implement greedy decoding for error
            loss = rnnt_loss(logits, y)
            error = 0

        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss, error

    for epoch in range(1, num_epochs+1):
        train_loss, train_error = 0
        for x, y, x_len, y_len in train_datset:
            loss, error = train_step(x, y, x_len, y_len)
            train_loss += loss
            train_error += error

            if step % 1000 == 0:
                dev_loss, dev_error, num_batch = 0, 0, 0
                for x, y, x_len, y_eln in dev_dataset:
                    loss, error = dev_step(x, y, x_len, y_len)
                    dev_loss += loss
                    dev_error += error
                    num_batch += 1
                print("Epoch %s, step %s, train loss %s, train error %s, dev loss %s, dev error %s" % 
                         (epoch, step, train_loss/step, train_error/step, dev_loss/num_batch, dev_error/num_batch))
                ckpt_manager.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ContextNet training module")

    # Joint network arguments
    parser.add_argument("--num_units", type=int, default=640, help="Joint network output size")
    parser.add_argument("--num_vocab", type=int, required=True, help="Output vocabulary size")

    # Label encoder arguments
    parser.add_argument("--num_lstms", type=int, default=2, help="Label encoder LSTM layers")
    parser.add_argument("--lstm_units", type=int, default=2048, help="Label encoder LSTM width")
    parser.add_argument("--out_dim", type=int, default=640, help="Label encoder output size")

    # Optimization arguments
    parser.add_argument("--lr", type=float, default=0.0025, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of epochs")
    parser.add_argument("--num_epoch", type=int, default=10, help="Number of epochs")

    # Train / validation data
    parser.add_argument("--data", type=str, required=True, help="Data directory having train/dev/test")
    parser.add_argument("--vocab", type=str, required=True, help="Vocabulary file")
    parser.add_argument("--mean", type=str, required=True, help="Mean file")
    parser.add_argument("--std_dev", type=str, required=True, help="Standard deviation file")
