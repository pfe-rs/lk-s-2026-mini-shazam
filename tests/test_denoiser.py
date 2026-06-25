import numpy as np
from pipeline.denoiser import NoDenoiser, UNetDenoiser


def test_no_denoiser_passthrough():
    nd = NoDenoiser()
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = nd.process(arr)
    assert np.array_equal(result, arr)


def test_unet_not_implemented():
    ud = UNetDenoiser()
    try:
        ud.process(np.array([[1.0]]))
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
