import numpy as np
import pytest
from eot.audio import last_window, log_mel
from eot.audio.frontend import _feature_extractor

@pytest.mark.parametrize('normalize',[False,True])
@pytest.mark.parametrize('length',[0,1600,16000,128000,256000])
def test_vectorized_frontend_matches_whisper(length,normalize):
    x=np.random.default_rng(31).normal(0,.03,length).astype(np.float32)
    expected=_feature_extractor()(last_window(x),sampling_rate=16000,return_tensors='np',
        padding='max_length',truncation=True,do_normalize=normalize)['input_features'][0]
    np.testing.assert_allclose(log_mel(x,normalize=normalize),expected,atol=3e-5,rtol=1e-5)
