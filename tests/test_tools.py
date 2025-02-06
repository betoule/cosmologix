from cosmologix.tools import load_csv_from_url


def test_csv():
    des_data = load_csv_from_url(
        "https://github.com/des-science/DES-SN5YR/raw/refs/heads/main/4_DISTANCES_COVMAT/DES-SN5YR_HD+MetaData.csv"
    )
    assert len(des_data) == 1829
    assert "zCMB" in des_data.dtype.names
