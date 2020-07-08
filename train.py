import argparse
import tensorflow as tf
from warprnnt_tensorflow import rnnt_loss

from dataset import create_dataset
from model import ConvBlock, ContextNet

# TODO SpecAugment

def create_conv_blocks():
    blocks = []

    # Kernel size is always 5
    # C0 : 1 conv layer, 256 output channels, strides 1, no residual
    blocks.append(ConvBlock([256//8, 256], 1, 256, 5, 1, residual=False))

    # C1-2 : 5 conv layers, 256 output channels, strides 1
    blocks.append(ConvBlock([256//8, 256], 5, 256, 5, 1))
    blocks.append(ConvBlock([256//8, 256], 1, 256, 5, 1))

    # C3 : 5 conv layers, 256 output channels, strides 2
    blocks.append(ConvBlock([256//8, 256], 5, 256, 5, 2))

    # C4-6 : 5 conv layers, 256 output channels, strides 1
    for i in range(4, 6+1):
        blocks.append(ConvBlock([256//8, 256], 5, 256, 5, 1))

    # C7 : 5 conv layers, 256 output channels, strides 2
    blocks.append(ConvBlock([256//8, 256], 5, 256, 5, 2))

    # C8-10 : 5 conv layers, 256 output channels, strides 1
    for i in range(8, 10+1):
        blocks.append(ConvBlock([256//8, 256], 5, 256, 5, 1))

    # C11-13 : 5 conv layers, 512 output channels, strides 1
    for i in range(11, 13+1):
        blocks.append(ConvBlock([512//8, 512], 5, 512, 5, 1))

    # C14 : 5 conv layers, 512 output channels, strides 2
    blocks.append(ConvBlock([512//8, 512], 5, 512, 5, 2))

    # C15-21 : 5 conv layers, 512 output channels, strides 1
    for i in range(15, 21+1):
        blocks.append(ConvBlock([512//8, 512], 5, 512, 5, 1))

    # C22 : 1 conv layers, 640 output channels, strides 1
    blocks.append(ConvBlock([640//8, 640], 5, 640, 5, 1, residual=False))

    return blocks

def create_model(**kwargs):
    kwargs["create_conv_blocks"] = create_conv_blocks
    return ContextNet(**kwargs)

def create_optimizer(lr):
    # TODO Transformer learning rate schedule
    return tf.keras.optimizers.Adam(learning_rate=lr)

def train(num_units, num_vocab, num_lstms, lstm_units, out_dim,
          lr, batch_size, num_epochs, data_path, vocab, mean, std_dev):
    model = create_model(num_units=num_units, num_vocab=num_vocab,
                         num_lstms=num_lstms, lstm_units=lstm_units, out_dim=out_dim)

    dev_dataset = create_dataset(data_path, "dev", vocab, mean, std_dev, batch_size)
    train_dataset = create_dataset(data_path, "train", vocab, mean, std_dev, batch_size)

    step = tf.Variable(1)
    optimizer = create_optimizer(lr)
    ckpt = tf.train.Checkpoint(step=step, optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, './ckpt', max_to_keep=10)

    # TODO Implement greedy decoding for error
    blank = num_vocab

    def dev_step(x, y, x_len, y_len):
        logits, x_len, y_len = model(x, y, x_len, y_len, training=False)
        if not tf.config.list_physical_devices('GPU'):
            logits = tf.nn.log_softmax(logits)
        loss = rnnt_loss(logits, y, x_len, y_len, blank)
        error = 0
        return tf.reduce_mean(loss), error

    def train_step(x, y, x_len, y_len):
        with tf.GradientTape() as tape:
            logits, x_len, y_len = model(x, y, x_len, y_len, training=True)
            if not tf.config.list_physical_devices('GPU'):
                logits = tf.nn.log_softmax(logits)
            loss = rnnt_loss(logits, y, x_len, y_len, blank)
            error = 0

        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return tf.reduce_mean(loss), error

    for epoch in range(1, num_epochs+1):
        train_loss, train_error, train_batches = 0, 0, 0
        for x, y, x_len, y_len in train_dataset:
            loss, error = train_step(x, y, x_len, y_len)
            train_loss += loss
            train_error += error
            train_batches += 1

            if step % 1000 == 0:
                dev_loss, dev_error, dev_batches = 0, 0, 0
                for x, y, x_len, y_len in dev_dataset:
                    loss, error = dev_step(x, y, x_len, y_len)
                    dev_loss += loss
                    dev_error += error
                    dev_batches += 1
                print("Epoch %s, step %s, train loss %s, train error %s, dev loss %s, dev error %s" % 
                         (epoch, step, train_loss/train_batches, train_error/train_batches,
                          dev_loss/dev_batches, dev_error/dev_batches))
                ckpt_manager.save()

            step.assign_add(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ContextNet training module")

    # Joint network arguments
    parser.add_argument("--num_units", type=int, default=640, help="Joint network output size")
    parser.add_argument("--num_vocab", type=int, required=True, help="Output vocabulary size")

    # Label encoder arguments
    parser.add_argument("--num_lstms", type=int, default=1, help="Label encoder LSTM layers")
    parser.add_argument("--lstm_units", type=int, default=2048, help="Label encoder LSTM width")
    parser.add_argument("--out_dim", type=int, default=640, help="Label encoder output size")

    # Optimization arguments
    parser.add_argument("--lr", type=float, default=0.0025, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")

    # Train / validation data
    parser.add_argument("--data_path", type=str, required=True, help="Data directory having train/dev/test")
    parser.add_argument("--vocab", type=str, required=True, help="Vocabulary file")
    parser.add_argument("--mean", type=str, required=True, help="Mean file")
    parser.add_argument("--std_dev", type=str, required=True, help="Standard deviation file")

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
