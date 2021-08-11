from dante_parser.data import ids_sents_train_test_split


def test_ids_sents_train_test_split():

    sent_ids = ["1", "2", "3", "4"]
    sent_texts = ["text1", "text2", "text3", "text4"]

    train_ids, test_ids, train_texts, test_texts = ids_sents_train_test_split(
        sent_ids, sent_texts, 0.2
    )

    assert len(train_ids) == len(train_texts)
    assert len(test_ids) == len(test_texts)
    assert all(x != y for x, y in zip(train_ids, test_ids))
