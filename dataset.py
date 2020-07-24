import glob
import os
import soundfile
import numpy as np
import tensorflow as tf

from utils import _get_audio_features_mfcc

# TODO BPE encoding
# TODO 80 dimensional logmel as input

def _get_output_sequence(vocab, transcript):
    if isinstance(transcript, bytes):
        transcript = transcript.decode('utf-8')
    labels = [vocab[char] if char in vocab else vocab['<unk>'] for char in transcript]
    return np.array(labels, dtype=np.int32), np.array(len(labels), dtype=np.int32)

def create_dataset(librispeech_dir, data_key, vocab, mean=None, std_dev=None, num_feats=40):
    """ librispeech_dir (str): path to directory containing librispeech data
        data_key (str) : train / dev / test
        mean (str|None) : path to file containing mean of librispeech training data
        std_dev (str|None) : path to file containing std_dev of librispeech training data
        num_feats (int) : input feature dimension

        Returns : tf.data.dataset instance
    """
    vocab = eval(open(vocab).read().strip())
    if mean:
        mean = np.loadtxt(mean).astype("float32")
    if std_dev:
        std_dev = np.loadtxt(std_dev).astype("float32")

    def _generate_librispeech_examples():
        """Generate examples from a Librispeech directory."""
        audios, transcripts = [], []
        transcripts_glob = os.path.join(librispeech_dir, "%s*/*/*/*.txt" % data_key)
        for transcript_file in glob.glob(transcripts_glob):
            path = os.path.dirname(transcript_file)
            for line in open(transcript_file).read().strip().splitlines():
                line = line.strip()
                key, transcript = line.split(" ", 1)
                audio_file = os.path.join(path, "%s.flac" % key)
                audios.append(audio_file)
                transcripts.append(transcript)
        return audios, transcripts

    def _extract_audio_features(audio_file):
        audio, sample_rate = soundfile.read(audio_file)
        feats = _get_audio_features_mfcc(audio, sample_rate)

        if mean is not None:
            feats = feats - mean
        if std_dev is not None:
            feats = feats / std_dev
        return feats, np.array(feats.shape[0], dtype=np.int32)

    def _extract_output_sequence(transcript):
        return _get_output_sequence(vocab, transcript)

    def _prepare(audio_file, transcript):
        audio_feats, timesteps = _extract_audio_features(audio_file)
        output_seq, seq_len = _extract_output_sequence(transcript)
        return audio_feats, output_seq, timesteps, seq_len

    audios, transcripts = _generate_librispeech_examples()
    dataset = tf.data.Dataset.from_tensor_slices((audios, transcripts))
    dataset = dataset.shuffle(1000000)

    dataset = dataset.map(lambda audio_file, transcript: \
                          tf.numpy_function(_prepare, [audio_file, transcript],
                          (tf.float32, tf.int32, tf.int32, tf.int32)),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Remove utterances which have >300 chars
    dataset = dataset.filter(lambda x, y, _, y_len: y_len <= 300)
    dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
                  element_length_func=lambda x, y, x_len, _: x_len,
                  bucket_boundaries=[500, 1000, 1250, 1500, 2000],
                  bucket_batch_sizes=[32, 16, 16, 8, 8, 4],
                  padded_shapes=([None, num_feats], [None], [], [])))
    return dataset
