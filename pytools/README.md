# Pytools

This is a collection of python notebooks and scripts designed to help with the launching and processing of the GPUIsing code.

Tools in general are softed into those used for making inputs in pyconf or visulising outputs in pyviz.
Tools that may be of general use such as grid_reader and grid_writer are exported from pytools directly.


## Python Envronment

We use [poetry](https://python-poetry.org/), the environment is defined in pyproject.toml
You can install the project dependancies using `poetry install`, [details here](https://python-poetry.org/docs/basic-usage/#installing-with-poetrylock)
Basic use is to use poetry shell or use the environment located at `~/.cache/pypoetry/virtualenvs/`.
Use `poetry env info` for more info

Note: 
> When you get the error `Failed to unlock the collection` you need to run:
> `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`
> See [SO](https://stackoverflow.com/questions/74438817/poetry-failed-to-unlock-the-collection)


There are two sides to the python:

## pyconf

Pyconf is a collection of helper scripts to create a launch configuration that maximises GPU use

TODO: More docs

## pyviz

Pyviz is a collection of helper scripts to load the output and visualise it

TODO: More Docs
