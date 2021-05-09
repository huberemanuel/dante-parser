# dante-parser
Automatic parsing of Brazilian Tweets for the stock market domain

## UDPipe

To run the UDPipe tagger (parser in the future), the SWIG is required, and can be installed using the following commands:

```bash
wget https://ufpr.dl.sourceforge.net/project/swig/swig/swig-4.0.2/swig-4.0.2.tar.gz -O swig.tar.gz
mkdir swig
tar -xvf swig.tar.gz --strip-components 1
cd swig
./configure
make
sudo make install
```

Download udpipe and create python bindings

```bash
make download-udpipe
make udpipe-bind
```

