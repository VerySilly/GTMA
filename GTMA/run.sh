#!/bin/bash

for i in $(seq 0 4);do
    echo =======================FOLD$i=======================
    python main.py --FF_number $i
done