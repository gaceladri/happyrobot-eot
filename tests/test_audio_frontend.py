import numpy as np
import pytest
from transformers import WhisperFeatureExtractor

from eot.audio import N_MELS, SAMPLE_RATE, WINDOW_SECONDS, last_window, log_mel


def _feature_extractor():
    return WhisperFeatureExtractor(chunk_length=int(WINDOW_SECONDS), feature_size=N_MELS, sampling_rate=SAMPLE_RATE)

@pytest.mark.parametrize('normalize',[False,True])
@pytest.mark.parametrize('length',[0,1600,16000,128000,256000])
def test_vectorized_frontend_matches_whisper(length,normalize):
    x=np.random.default_rng(31).normal(0,.03,length).astype(np.float32)
    expected=_feature_extractor()(last_window(x),sampling_rate=16000,return_tensors='np',
        padding='max_length',truncation=True,do_normalize=normalize)['input_features'][0]
    np.testing.assert_allclose(log_mel(x,normalize=normalize),expected,atol=3e-5,rtol=1e-5)
