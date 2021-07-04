import os
import tempfile

from dante_parser.data.conllu import read_conllu


def test_read_conlly():

    # Create dummy conllu
    fd, f_name = tempfile.mkstemp()

    with os.fdopen(fd, "w") as tmp:
        tmp.write("# newpar\n")
        tmp.write("# newdoc id = id1\n")
        tmp.write("# sent_id = 1\n")
        tmp.write("# text = teste\n")
        tmp.write("1\tteste_\t_\t_\t_\t_\t_\t_\tSpacesAfter=\\n\n\n")
        tmp.write("# newpar\n")
        tmp.write("# newdoc id = id2\n")
        tmp.write("# sent_id = 2\n")
        tmp.write("# text = teste dois\n")
        tmp.write("1\tteste_\t_\t_\t_\t_\t_\t_\t_\n")
        tmp.write("2\tdois_\t_\t_\t_\t_\t_\t_\tSpacesAfter=\\n\n\n")

    sents = read_conllu(f_name)
    assert len(sents) == 2
