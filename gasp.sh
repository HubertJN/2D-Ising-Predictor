#!/bin/bash

# Tools to make it easier for users to get started with gasp

# Command line arguments
# -h, --help: print help message
# -v, --version: print version number
# --python_init: create the poetry virtual environment
# --pytest: run pytest
# --make: run make
# --run: run the program



POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      echo "Usage: ./gasp.sh [options]"
      exit
      ;;
    -v|--version)
      cat VERSION
      exit
      ;;
    --python_init)
      cd pytools/
      poetry shell
      cd ..
      exit # past argument
      ;;
    --pytest)
      pytest
      exit
      ;;
    --test_logfile)
      python3 invoke_pytest.py $2
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
      bin/gasp $2
      exit
      ;;
    --debug) $2
      make debug
      echo "Running cuda-gdb bin/gasp_debug, once in the debugger type \`run" $2"\` to run the program"
      cuda-gdb bin/gasp_debug
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
