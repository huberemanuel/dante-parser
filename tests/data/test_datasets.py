from dante_parser.data import get_datasets


def test_get_datasets():

    datasets = get_datasets()

    assert len(datasets.keys()) > 0

    for key, data in datasets.items():
        for set_key, set_value in data.items():

            assert "path" in set_value.keys()
            assert "filetype" in set_value.keys()
            assert set_value["filetype"] in ["csv", "conllu"]
