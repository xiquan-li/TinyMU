import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from einops import rearrange
from timm.models.vision_transformer import Block
from functools import partial
import torchaudio


# from matpac.preprocess import logMelSpectrogram
# from matpac.encoder import encoder_layers_config, encoder_layers
# from matpac.utils import PatchEmbed

def expand_size(sz):
  if isinstance(sz, int):
    return [sz, sz]
  return sz

class PatchEmbed(nn.Module):
  """ 2D Image to Patch Embedding -- borrowed from https://pypi.org/project/timm/0.4.12/
  """

  def __init__(self, img_size=[80, 304], patch_size=16, in_chans=1, embed_dim=768, norm_layer=None, flatten=True):
    super().__init__()
    img_size = expand_size(img_size)
    patch_size = expand_size(patch_size)
    self.img_size = img_size
    self.patch_size = patch_size
    self.grid_size = (img_size[0] // patch_size[0],
                      img_size[1] // patch_size[1])
    self.num_patches = self.grid_size[0] * self.grid_size[1]
    self.flatten = flatten

    self.proj = nn.Conv2d(in_chans, embed_dim,
                          kernel_size=patch_size, stride=patch_size)
    self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

  def forward(self, x):
    x = self.proj(x)
    if self.flatten:
      x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
    x = self.norm(x)
    return x


@dataclass
class encoder_layers_config:
  embed_dim: int = 768
  depth: int = 12
  num_heads: int = 12
  mlp_ratio: int = 4


class encoder_layers(nn.Module):
  """Vision transformer encoder layers.
  """

  def __init__(self,
               cfg: encoder_layers_config):
    """
      Parameters
      ----------
      cfg : encoder_layers_config
          dataclass with all the parameters for the transformer layers and the 
          layer norm.
    """
    super().__init__()
    self.blocks = nn.ModuleList([
        Block(cfg.embed_dim,
              cfg.num_heads,
              cfg.mlp_ratio,
              qkv_bias=True,
              norm_layer=partial(nn.LayerNorm, eps=1e-6))
        for i in range(cfg.depth)])
    self.norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)

  def forward(self, x,
              return_layers=False):
    """Forward of a sequence through the transformer layers

    Parameters
    ----------
    x : torch.tensor
        the sequence to pass through the layers
    return_layers : bool, optional
        If true returns the output of each layers, by default False

    Returns
    -------
    torch.tensor
        Either the output of the last layer, or the stacked output of each
        layer.
    """

    layers = []
    for blk in self.blocks:
      x = blk(x)
      if return_layers:
        layers.append(x.unsqueeze(dim=1))
    x = self.norm(x)
    if return_layers:
      # replace the last feature with the normalized one.
      layers[-1] = x.unsqueeze(dim=1)
      return torch.cat(layers, dim=1)
    else:
      return x
    

class logMelSpectrogram(torch.nn.Module):
  r"""Create MelSpectrogram for a raw audio signal. Wrapping of torchaudio class, but
  with parameters from librosa and time input in seconds for convenience.

  .. devices:: CPU CUDA

  .. properties:: Autograd TorchScript

  This is a composition of :py:func:`torchaudio.transforms.Spectrogram` and
  and :py:func:`torchaudio.transforms.MelScale`.

  Sources
      * https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
      * https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
      * http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

  Args:
      sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
      n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
      win_length (float or None, optional): Window size. (Default: ``n_fft``)
      hop_length (float or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
      f_min (float, optional): Minimum frequency. (Default: ``0.``)
      f_max (float or None, optional): Maximum frequency. (Default: ``None``)
      pad (int, optional): Two sided padding of signal. (Default: ``0``)
      n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
      window_fn (Callable[..., Tensor], optional): A function to create a window tensor
          that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
      power (float, optional): Exponent for the magnitude spectrogram,
          (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
      normalized (bool, optional): Whether to normalize by magnitude after stft. (Default: ``False``)
      wkwargs (Dict[..., ...] or None, optional): Arguments for window function. (Default: ``None``)
      center (bool, optional): whether to pad :attr:`waveform` on both sides so
          that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
          (Default: ``True``)
      pad_mode (string, optional): controls the padding method used when
          :attr:`center` is ``True``. (Default: ``"reflect"``)
      onesided: Deprecated and unused.
      norm (str or None, optional): If "slaney", divide the triangular mel weights by the width of the mel band
          (area normalization). (Default: ``None``)
      mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

  Example
      >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
      >>> transform = transforms.MelSpectrogram(sample_rate)
      >>> mel_specgram = transform(waveform)  # (channel, n_mels, time)

  See also:
      :py:func:`torchaudio.functional.melscale_fbanks` - The function used to
      generate the filter banks.
  """

  def __init__(self,
               sample_rate=16000,
               n_fft=None,
               win_length=0.025,
               hop_length=0.01,
               f_min=0.0,
               f_max=None,
               log_offset=0.001,
               pad=0,
               n_mels=128,
               window_fn=torch.hann_window,
               power=2.0,
               normalized=False,
               wkwargs=None,
               center=False,
               pad_mode="reflect",
               onesided=None,
               norm="slaney",
               mel_scale="slaney",
               ) -> None:
    super(logMelSpectrogram, self).__init__()

    if f_max is None:
      f_max = sample_rate // 2

    win_length = int(np.round(sample_rate * win_length))

    if n_fft is None:
      n_fft = win_length

    if hop_length is None:
      hop_length = win_length // 2
    else:
      hop_length = int(np.round(sample_rate * hop_length))

    self.MelSpectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        pad=pad,
        n_mels=n_mels,
        window_fn=window_fn,
        power=power,
        normalized=normalized,
        wkwargs=wkwargs,
        center=center,
        pad_mode=pad_mode,
        onesided=onesided,
        norm=norm,
        mel_scale=mel_scale)

    self.log_offset = log_offset

  def forward(self, waveform):
    mel_specgram = self.MelSpectrogram(waveform)
    log_melspecgram = torch.log(mel_specgram + self.log_offset)
    return log_melspecgram
  
  
@dataclass
class general_config:

  ## Model Parameters ##
  encoder: encoder_layers_config = field(default_factory=encoder_layers_config)

  ## Logmel Spec Shape ##
  n_freq: int = 80
  n_t: int = 992  # 修改以匹配checkpoint: 992//16=62, 5*62=310, +1=311
  patch_size: int = 16

  ## Audio Parameters ##
  sr: int = 16000

  ## Normalization params ##
  lms_mean: float = -7.056
  lms_std: float = 4.193


class matpac_wrapper(nn.Module):
  def __init__(self,
               inference_type="precise",  # or fast
               pull_time_dimension=True,
               cfg: general_config = general_config(),
               ):
    super(matpac_wrapper, self).__init__()

    # Setting the parameters
    self.cfg = cfg
    self.inference_type = inference_type
    self.pull_time_dimension = pull_time_dimension

    # Setting the nn modules
    self.patch_embed = PatchEmbed(img_size=[cfg.n_freq, cfg.n_t],
                                  patch_size=cfg.patch_size,
                                  embed_dim=cfg.encoder.embed_dim,
                                  )

    num_patches = self.patch_embed.num_patches

    self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.encoder.embed_dim))
    self.pos_embed = nn.Parameter(
        torch.zeros(
            1, num_patches + 1, cfg.encoder.embed_dim),
        requires_grad=False
    )

    self.student_encoder = encoder_layers(cfg=cfg.encoder)

    # Setting the forward tools for pre-processing audios
    self.log_mel = logMelSpectrogram(sample_rate=cfg.sr,
                                     n_fft=400,
                                     win_length=0.025,
                                     hop_length=0.01,
                                     f_min=50,
                                     f_max=cfg.sr//2,
                                     log_offset=torch.finfo().eps,
                                     n_mels=cfg.n_freq,
                                     center=False)

  def preprocess(self, x):

    if x.ndim < 2:
      x = x.unsqueeze(dim=0)

    x = self.log_mel(x)

    x = (x - self.cfg.lms_mean) / self.cfg.lms_std

    return x

  def forward(self, x):

    x = self.preprocess(x)

    # TODO DELETE
    if x.ndim > 3:
      x = x.squeeze(dim=1)

    if self.inference_type == "precise":
      emb, layer_results = self.forward_precise(x)

    elif self.inference_type == "fast":
      emb, layer_results = self.forward_fast(x)

    if self.pull_time_dimension:
      emb = emb.mean(dim=1)
      layer_results = layer_results.mean(dim=2)

    return emb, layer_results

  def forward_fast(self, x):
    """Forward when we want to extract efficiently, better for finetuning,
    the forward is faster but adds a lot of padding sometimes compared to the
    precise forward.

    Parameters
    ----------
    x : torch.tensor
        of shape [bs, n_samples]

    Returns
    -------
    emb
        the embedding of the input audio of shape [bs, time, emb_dim]
    layer_results
        the output of each layer with shape [bs, n_layers, time, emb_dim]
    """

    bs, _, _ = x.shape

    patch_fbins = self.grid_size()[0]
    unit_frames = self.cfg.n_t
    embed_d = self.patch_embed.proj.out_channels

    n_chunk = (x.shape[-1] + unit_frames - 1) // unit_frames
    pad_frames = n_chunk*unit_frames - x.shape[-1]

    if pad_frames > 0:
      x = torch.nn.functional.pad(x, (0, pad_frames))

    x_full = rearrange(x, 'b f (n u) -> (b n) f u',
                       n=n_chunk, f=x.shape[-2], b=bs)

    _, layer_results_full = self.extract_features(
        x_full.unsqueeze(dim=1))

    layer_results_full = layer_results_full[..., 1:, :]

    layer_results_full = rearrange(layer_results_full, 'b l (f t) d -> b l t (f d)',
                                   f=patch_fbins, d=embed_d)

    layer_results_full = rearrange(layer_results_full, '(b n) l t d -> b l (t n) d',
                                   b=bs, n=n_chunk, d=embed_d*patch_fbins)

    emb = layer_results_full[:, -1]

    return emb, layer_results_full

  def forward_precise(self, x):
    """Forward that is precise but might be slow on big batches,
    this is the forward used for the results in the paper

    Parameters
    ----------
    x : torch.tensor
        of shape [bs, n_samples]

    Returns
    -------
    emb
        the embedding of the input audio of shape [bs, time, emb_dim]
    layer_results
        the output of each layer with shape [bs, n_layers, time, emb_dim]
    """

    patch_fbins = self.grid_size()[0]
    unit_frames = self.cfg.n_t
    patch_frames = self.patch_embed.patch_size[1]
    embed_d = self.patch_embed.proj.out_channels
    n_chunk = (x.shape[-1] + unit_frames - 1) // unit_frames
    pad_frames = (patch_frames - x.shape[-1] % patch_frames) % patch_frames
    if pad_frames > 0:
      x = torch.nn.functional.pad(x, (0, pad_frames))

    x = x.unsqueeze(dim=1)
    embeddings = []
    for i in range(n_chunk):
      emb, layer_results = self.extract_features(
          x[..., i*unit_frames:(i+1)*unit_frames])

      layer_results = layer_results[..., 1:, :]
      layer_results = rearrange(layer_results, 'b n (f t) d -> b n t (f d)',
                                f=patch_fbins, d=embed_d)
      embeddings.append(layer_results)
    layer_results = torch.cat(embeddings, axis=-2)

    layer_results = layer_results
    emb = layer_results[:, -1]

    return emb, layer_results

  def extract_features(self, x):
      # Patch Logmel Spectrogram
    if x.ndim <= 3:
      x = x.unsqueeze(dim=1)
    x = self.patch_embed(x)

    # Add positional
    pos_embed = self.pos_embed[:, 1:, :]
    if x.shape[1] < pos_embed.shape[1]:
      # audio: shorten pos_embed for a short input
      dims = pos_embed.shape[-1]
      fbins = self.grid_size()[0]
      frames = x.shape[1] // fbins
      pos_embed = pos_embed.reshape(
          1, fbins, -1, dims)[:, :, :frames, :].reshape(1, fbins*frames, dims)
    x = x + pos_embed

    # append cls token
    cls_token = self.cls_token + self.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    x = self.student_encoder.forward(x, return_layers=True)

    return x[:, -1, :, :], x  # Return embedding and layer results

  def grid_size(self):
    # This fails with timm 0.4.5 -> return self.patch_embed.grid_size
    # Workaround for avoid compatibility issue
    img_size = np.array(self.patch_embed.img_size)
    patch_size = np.array(self.patch_embed.patch_size)
    grid_size = img_size // patch_size
    return grid_size


def get_matpac(checkpoint_path,
               inference_type: str = "precise",  # or fast
               pull_time_dimension: bool = True,
               config: general_config = None):
  """Basic function to instantiate the inference model from the paper
  with the best checkpoint.

  Parameters
  ----------
  checkpoint_path : str
      The path to the checkpoint
  inference_type : str
      We have to type of inference. The main difference is the speed and its 
      precision. 
      With "precise" we use all the audio without adding some padding, but it 
      is slow on big batches as it rely on a loop.
      With "fast" we do some padding on the audio and we do not have a loop 
      to extract the features, it is faster but less precise.
  pull_time_dimension : bool
      Parameter to decide if we pull the time dimension during inference or not.
      If True we take the mean of the time dimension.


  Returns
  -------
  _type_
      _description_
  """

  # Use provided config or default
  if config is None:
    config = general_config()
  
  model = matpac_wrapper(inference_type=inference_type,
                         pull_time_dimension=pull_time_dimension,
                         cfg=config)

  checkpoint = torch.load(checkpoint_path)

  # Load the model state dict from the checkpoint into the model
  model.load_state_dict(checkpoint, strict=False)

  model.eval()

  return model


if __name__ == "__main__":
  ckpt_path = "./weights/matpac_plus_as_48_1_map_enconly.pt"
  model = get_matpac(checkpoint_path=ckpt_path, pull_time_dimension=False)

  audio = torch.rand((1, 160000))
  audio_2 = torch.rand((2, 320000))

  print(model(audio))
