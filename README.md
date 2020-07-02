ContextNet: Improving Convolutional Neural Networks for Automatic Speech Recognition with Global Context

This repository contains TF2.x based implementation for [this paper](https://arxiv.org/pdf/2005.03191.pdf).

TODO:
  * Decoder is used at multiple places. What is it exactly?
  * Add details for prediction network and joint network

Dataset: 960 hours of [LibriSpeech](http://www.openslr.org/12)  
Input: Sequence of 80-dimensional filterbank features using 25msec window length and 10msec stride  
Output: 1K WPM

Optimization details:
  * [Adam](https://arxiv.org/abs/1412.6980) optimizer
  * Transformer LR schedule with 15K warmup steps and peak LR 0.0025
  * L2 regularization on all trainable weights
  * Variational noise added to decoder for regularization

Architecuture details:
  * RNN-Transducer based architecuture
  * Acoustic encoder is proposed in the paper, prediction network and joint network is based on LSTM layers as used in [this paper](https://arxiv.org/abs/1811.06621)
  * Acoustic encoder:
    It consists of multiple convolution blocks (23 in all the experiments), where each block C<sub>i</sub> is made up of multiple depth-wise separable convolution layers. A convolution block is shown below with details for all 23 blocks in following table.
    ![alt text](convblock.png) 

    | Block Id                     | #Conv Layers | #Output Channels | Kernel Size | Other       |
    |------------------------------|--------------|------------------|-------------|-------------|
    |C<sub>0</sub>                 | 1            | 256 x \alpha     | 5           | No residual |
    |C<sub>1</sub>-C<sub>2</sub>   | 5            | 256 x \alpha     | 5           |             |
    |C<sub>3</sub>                 | 5            | 256 x \alpha     | 5           | Stride is 2 |
    |C<sub>4</sub>-C<sub>6</sub>   | 5            | 256 x \alpha     | 5           |             |
    |C<sub>7</sub>                 | 5            | 256 x \alpha     | 5           | Stride is 2 |
    |C<sub>8</sub>-C<sub>10</sub>  | 5            | 256 x \alpha     | 5           |             |
    |C<sub>11</sub>-C<sub>13</sub> | 5            | 512 x \alpha     | 5           |             |
    |C<sub>14</sub>                | 5            | 512 x \alpha     | 5           | Stride is 2 |
    |C<sub>15</sub>-C<sub>21</sub> | 5            | 512 x \alpha     | 5           |             |
    |C<sub>22</sub>                | 1            | 640 x \alpha     | 5           | No residual |
    Stride of 2 in a convolution block means last convolution layer in that block has a stride of 2, rest of them have stride of 1.

    SE is squeeze and excitation layer as shown below
    ![alt text](SE.png) 

    3 different model variations with *global context* are shown below. The authors also experiment with context sizes of None, 256, 512 and 1024. Currently, the implementation allows either global context or no context at all.
    | Model | \alpha | #Params(M) |
    |-------|--------|------------|
    | Small | 0.5    | 10.8       |
    | Medium| 1.0    | 31.4       |
    | Large | 2.0    | 112.7      |

  * Decoder: Single layer LSTM with input dimension 640
  * (Optional/Not implemented) RNN-LM: 3 LSTM layers of width 4096

SpecAugment:
  * Mask parameter F = 27
  * 10 time masks with maximum time-mask ratio, p<sub>s</sub> = 0.05
  * Maximum size of the time mask p<sub>s</sub> * length-of-utterance
  * Time warping is not used

Author's observations:
  * Swish activation, f(x) = x \cdot \sigma ( \beta x), works better than RELU. \beta = 1 is used in the paper
  * Increasing context size improves the model performance. Model without any context also performs very well and is comparable with model performances with non-zero context size
  * Context in SE layer significantly improves performance on the test-other set
  * The proposed architecture is also effective on large scale dataset
  * A progressive downsampling of 8 achieves good tradeoff between computational cost and model performance
