#!/bin/bash

#check if the user has provided a run number
if [ -z "$1" ]
then
    echo "Please provide a run number"
    exit 1
fi




run_num=$1

scancel $run_num
./delold.sh $run_num

sbatch run.sh