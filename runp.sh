#!/usr/bin/env bash
# June 22, 2019
# Author: V. Mullachery
#
# Usage: nohup runp.sh > outputs/pgradrun.out 2>&1 &
#
mkdir -p outputs/
> outputs/pgradrun.out

mkdir -p models/

# source the Conda environment
source activate tfkerasenv

#run and collect log
python pgrad_agent.py > outputs/pgradrun.out
