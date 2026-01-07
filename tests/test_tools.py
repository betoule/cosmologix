from cosmologix.tools import load_csv_from_url, clear_cache, conflevel_to_delta_chi2
from numpy.testing import assert_array_equal, assert_allclose
import scipy.stats


def test_csv():
    clear_cache()
    des_data1 = load_csv_from_url(
        "https://raw.githubusercontent.com/des-science/DES-SN5YR/refs/tags/v1.2/4_DISTANCES_COVMAT/DES-SN5YR_HD%2BMetaData.csv"
    )
    des_data2 = load_csv_from_url(
        "https://raw.githubusercontent.com/des-science/DES-SN5YR/refs/tags/v1.2/4_DISTANCES_COVMAT/DES-SN5YR_HD%2BMetaData.csv"
    )
    assert len(des_data1) == 1829
    assert "zCMB" in des_data1.dtype.names
    assert_array_equal(des_data2, des_data1)


def test_conf_level():
    for level in 68, 95:
        assert_allclose(
            conflevel_to_delta_chi2(level, 2), scipy.stats.chi2.ppf(level / 100, 2)
        )
