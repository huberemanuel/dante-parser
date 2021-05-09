from ufal.udpipe import Trainer, InputFormat, Sentence

def format_sents(sents: list, input_format: InputFormat) -> list:
    """
    Converts conllu sentences to UDPipe Sentence

    Parameters
    ----------
    sents: list
        List of conllu formatted sentences
    input_format: InputFormat
        InputFormat being used.

    Returns
    -------
    list:
        List of Sentence objects
    """
    out_sents = []

    for sent in sents:
        input_format.setText(sent)
        s = Sentence()
        input_format.nextSentence(s)
        out_sents.append(s)

    return out_sents


def train_udpipe(train_sents: list, val_sents: list, model_name: str):
    """
    Calls UDPipe training routines

    Parameters
    ----------
    train_sents: list
        List of conllu formatted sentences
    val_sents: list
        List of conllu formatted sentences
    model_name: str
        Name of the output model
    """
    input_format = InputFormat.newConlluInputFormat()
    train_sents = format_sents(train_sents, input_format)
    val_sents = format_sents(val_sents, input_format)

    trainer = Trainer()
    model = trainer.train("morphodita_parsito", train_sents, val_sents, 
                          Trainer.DEFAULT, Trainer.DEFAULT, Trainer.DEFAULT)
    
    model_file = open(model_name, "wb")
    model_file.write(model.encode("utf-8", errors="surrogateescape"))
    model_file.close()

