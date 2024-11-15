#!/bin/bash


usage() {
    echo "Usage: delold.sh <run_num> [-model_name <model_name>] [-data_dir <data_dir>]"
    exit 1
}


if [ "$#" -ne 1 ] && [ "$#" -ne 3 ] && ["$#" -ne 5]; then
    usage
fi

run_num=$1
model_name="model.pth"
data_dir="data"


if [ "$#" -eq 3 ]; then
    if [ "$2" == "-model_name" ]; then
        model_name=$3
    elif [ "$2" == "-data_dir" ]; then
        data_dir=$3
    else
        usage
    fi
fi

if [ "$#" -eq 5 ]; then
    if [ "$2" == "-model_name" ]; then
        model_name=$3
        if [ "$4" == "-data_dir" ]; then
            data_dir=$5
        else
            usage
        fi
    elif [ "$2" == "-data_dir" ]; then
        data_dir=$3
        if [ "$4" == "-model_name" ]; then
            model_name=$5
        else
            usage
        fi
    else
        usage
    fi
fi


scancel $run_num
mv $model_name models/model_$run_num.pth



zip run_$run_num.zip "$data_dir"/* slurm-$run_num.out models/model_$run_num.pth
rm "$data_dir"/* slurm-$run_num.out 
mv run_$run_num.zip oldruns/run_$run_num.zip
