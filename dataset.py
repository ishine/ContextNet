import glob
import os
import soundfile
import numpy as np
import tensorflow as tf

from utils import _get_audio_features_mfcc

########## Dependencies ##############
# soundfile : pip install pysoundfile
######################################

# TODO BPE encoding
def _get_output_sequence(vocab, transcript):
    labels = [vocab[char] for char in transcript]
    return np.array(labels, dtype="int32"), len(labels)

def create_dataset(librispeech_dir, data_key, vocab, mean=None, std_dev=None, batch_size=1):
    """ librispeech_dir (str): path to directory containing librispeech data
        data_key (str) : train / dev / test
        mean (str|None) : path to file containing mean of librispeech training data
        std_dev (str|None) : path to file containing std_dev of librispeech training data

        Returns : tf.data.dataset instance
    """
    vocab = eval(open(vocab).read().strip())

    if mean:
        mean = np.loadtxt(mean).astype("float32")
    if std_dev:
        std_dev = np.loadtxt(std_dev).astype("float32")

    def _generate_librispeech_examples():
      """Generate examples from a Librispeech directory."""
      transcripts_glob = os.path.join(librispeech_dir, "%s/*/*/*.txt" % data_key)
      for transcript_file in glob.glob(transcripts_glob):
          path = os.path.dirname(transcript_file)
          for line in open(transcript_file).read().strip().splitlines():
              line = line.strip()
              key, transcript = line.split(" ", 1)
              audio_file = os.path.join(path, "%s.flac" % key)
              yield key, audio_file, transcript

    def _extract_audio_features(audio_file):
        audio, sample_rate = soundfile.read(audio_file)
        feats = _get_audio_features_mfcc(audio, sample_rate)

        if mean is not None:
            feats = feats - mean
        if std_dev is not None:
            feats = feats / std_dev
        return feats, feats.shape[0]

    def _extract_output_sequence(transcript):
        return _get_output_sequence(vocab, transcript)

    def _prepare(audio_file, transcript):
        audio_feats, timesteps = _extract_audio_features(audio_file)
        output_seq, seq_len = _extract_output_sequence(transcript)
        return audio_feats, output_seq, timesteps, seq_len

    dataset = tf.data.Dataset.from_generator(_generate_librispeech_examples,
                                             (tf.string, tf.string, tf.string))
    dataset = dataset.map(lambda _, audio_file, transcript: _prepare(audio_file, transcript),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, num_features], [None]))
    return dataset
