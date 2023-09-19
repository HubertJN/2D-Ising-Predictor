#!/usr/bin/env bash

# Either get current git tag if inside a repo, otherwise fall back
# to alternative. Use a 10 char limit

# Check if this is a git repo - grab ten char (min) commit ID if we are
commit=`git rev-parse --short=10 HEAD 2> /dev/null`

if [ $? -eq 0 ]
then
  version=`git describe --tags 2>/dev/null`  # Get current tag - swallow errors
  if [ $? -ne 0 ]
  then
    version=$commit # No tags - fall back to commit id
  fi
else
  version=`date -Idate` #Not a git repo - fall back to current date
fi

echo VERSION=$version | cut -c-18 > VERSION
