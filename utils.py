# Taken from https://github.com/rwth-i6/returnn/blob/master/GeneratingDataset.py
def _get_audio_features_mfcc(audio, sample_rate, window_len=0.025, step_len=0.010, num_feature_filters=40):
  """
  :param numpy.ndarray audio: raw audio samples, shape (audio_len,)
  :param int sample_rate: e.g. 22050
  :param float window_len: in seconds
  :param float step_len: in seconds
  :param int num_feature_filters:
  :return: (audio_len // int(step_len * sample_rate), num_feature_filters), float32
  :rtype: numpy.ndarray
  """
  import librosa
  mfccs = librosa.feature.mfcc(
    audio, sr=sample_rate,
    n_mfcc=num_feature_filters,
    hop_length=int(step_len * sample_rate), n_fft=int(window_len * sample_rate))
  librosa_version = librosa.__version__.split(".")
  if int(librosa_version[0]) >= 1 or (int(librosa_version[0]) == 0 and int(librosa_version[1]) >= 7):
    rms_func = librosa.feature.rms
  else:
    rms_func = librosa.feature.rmse
  energy = rms_func(
    audio,
    hop_length=int(step_len * sample_rate), frame_length=int(window_len * sample_rate))
  mfccs[0] = energy  # replace first MFCC with energy, per convention
  assert mfccs.shape[0] == num_feature_filters  # (dim, time)
  mfccs = mfccs.transpose().astype("float32")  # (time, dim)
  return mfccs
