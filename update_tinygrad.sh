#!/bin/bash

rm -rf tinygrad
git clone git@github.com:tinygrad/tinygrad.git --depth 1
mv tinygrad/tinygrad tmp
rm -rf tinygrad
mv tmp tinygrad
