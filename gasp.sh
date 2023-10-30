# Tools to make it easier for users to get started with gasp

# Command line arguments
# -h, --help: print help message
# -v, --version: print version number
# --python_init: create the poetry virtual environment
# --pytest: run pytest
# --make: run make
# --run: run the program

#!/bin/bash

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      echo "Usage: gasp.sh [options]"
      exit
      ;;
    -v|--version)
      cat VERSION
      exit
      ;;
    --python_init)
      cd pytools/
      poetry shell
      cd -
      exit # past argument
      ;;
    --pytest)
      pytest
      exit
      ;;
    --make_config)
      python3 pytools/pyconf/main.py
      exit
      ;;
    --make)
      make $2
      exit
      ;;
    --run)
      bin/gasp /configurations/$2
      exit
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done
