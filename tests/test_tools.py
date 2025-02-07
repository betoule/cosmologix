from cosmologix.tools import load_csv_from_url, clear_cache
from numpy.testing import assert_array_equal


def test_csv():
    clear_cache()
    des_data1 = load_csv_from_url(
        "https://github.com/des-science/DES-SN5YR/raw/refs/heads/main/4_DISTANCES_COVMAT/DES-SN5YR_HD+MetaData.csv"
    )
    des_data2 = load_csv_from_url(
        "https://github.com/des-science/DES-SN5YR/raw/refs/heads/main/4_DISTANCES_COVMAT/DES-SN5YR_HD+MetaData.csv"
    )
    assert len(des_data1) == 1829
    assert "zCMB" in des_data1.dtype.names
    assert_array_equal(des_data2, des_data1)
