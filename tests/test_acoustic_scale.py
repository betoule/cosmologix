from cosmologix.acoustic_scale import rs, z_star, z_drag
from cosmologix import Planck18
import pyccl as ccl


def test_acoustic_scale():
    assert abs(z_star(Planck18) - 1091.73) < 1e-2
    assert abs(z_drag(Planck18) - 1020.715) < 1e-2
    assert abs(rs(Planck18, z_star(Planck18)) - 144637.42429246) < 1e-3
