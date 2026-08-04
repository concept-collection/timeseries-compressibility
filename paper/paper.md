# How close are lossless codecs to the entropy rate of extracellular recordings and other noisy quantized signals?

## Abstract

Storage is a real cost in extracellular electrophysiology: a 512-channel probe sampled at 30 kHz produces 1.8 GB per minute, and recordings can run for hours. Compression methods are usually chosen by benchmarking candidates against one another, not by reference to theory or to any absolute limit on achievable rate. Yet most of what must be encoded is noise, which is well modeled as a stationary Gaussian process observed through a quantizer. We give a consistent Monte Carlo estimator of the model's entropy rate — the bound no lossless codec can beat — together with an analytic approximation that depends on the power spectrum alone. The fitted model predicts the rates achieved by every prediction-based codec we test to within XX bits/sample. Measured against this bound, the codecs in conventional use fall well short (XX–XX%). Linear predictive coding followed by entropy coding with Asymmetric Numeral Systems comes within 5–10% of the limit, depending on the recording setup and on the preprocessing applied before compression. We establish these results on synthetic sources as well as on three real electrophysiology recordings, raw and bandpass-filtered.

# Introduction

# Setup


