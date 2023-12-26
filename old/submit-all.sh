#!/bin/bash

for script in $(ls *.slurm)
do
	sbatch $script
done
