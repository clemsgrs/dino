#!/bin/sh

if [ -n "$1" ]; then
  echo "linting \"$1\""
fi

echo "running black"
if [ -n "$1" ]; then
  black "$1"
else
  black dino
fi

echo "running flake8"
if [ -n "$1" ]; then
  flake8 "$1"
else
  flake8 --ignore=E501,F401,W503
fi

echo "running pylint"
if [ -n "$1" ]; then
  pylint "$1"
else
  pylint dino
fi

exit 0
