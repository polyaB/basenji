#!/bin/sh

wget https://storage.googleapis.com/basenji_barnyard/model_human.h5
wget https://storage.googleapis.com/basenji_barnyard/model_mouse.h5

# also have .tf but they are directories, so easier to use gsutil cp -r
