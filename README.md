# GASP

## For Users:

### Setup

First make gasp.sh an executable, `chmod a+x gasp.sh`

The gasp launching tool can now be used as `./gasp.sh [option]`

Make the python envronment `./gasp.sh --python_init`, this launches a poetry virtualenvronment.

Test the python scripts `./gasp.sh --pytest`, runs the pytest suit.

Build the GASP GPU code `./gasp.sh --make` or `make`.

### Configure

To enter the GASP configurator tool run `./gasp.sh --make_config`

GASP configuration tool information can be found [here](./pytools/README.md)

