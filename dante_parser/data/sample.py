from sklearn.model_selection import train_test_split

def ids_sents_train_test_split(sent_ids: list, sent_texts: list, test_size: float = 0.2,
                               random_state: int=42):
    """
    Split dataset into train and test sets. 

    Parameters
    ----------
    sent_ids: list
        List of sentences ids
    sent_texts: list
        List of sentence tokens
    test_size: float
        Proportion of data to use on test set.

    Returns
    -------
    list:
        Shuffled train sentence ids
    lsit:
        Shuffled train sentence 
    list:
        Shuffled test sentence ids
    lsit:
        Shuffled test sentence 
    """

    if len(sent_ids) != len(sent_texts):
        raise ValueError("sent_ids and sent_texts length must match!")

    train_ids, test_ids, train_texts, test_texts = train_test_split(sent_ids, sent_texts, 
                                                                       test_size=test_size, 
                                                                       random_state=random_state)

    return train_ids, test_ids, train_texts, test_texts

def sents_train_test_split(sent_texts: list, test_size: float=0.2, random_state: int=42):
    train, test = train_test_split(sent_texts, test_size=test_size, random_state=random_state)
    return train, test
