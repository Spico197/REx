#!/bin/bash

set -vx

cd raw

unzip KCT_test_public.zip
unzip KCT_train_public.zip

# shuffle
shuf -o train_shuffled.txt KCT_train_public.txt

# split
NUM_LINE=$(wc -l train_shuffled.txt | cut -d ' ' -f 1)
NUM_UNIT=$(( NUM_LINE / 10 ))
split -l $NUM_UNIT -e -d 'train_shuffled.txt' seg
mv seg00 dev.txt
mv seg01 test.txt
cat seg* > train.txt
rm seg*

# move
cd ..
mkdir formatted
mv raw/train.txt raw/dev.txt raw/test.txt formatted
cp raw/KCT_test_public.txt formatted/submit1.txt
