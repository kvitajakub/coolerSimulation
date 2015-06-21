#!/bin/bash
mpirun --prefix /usr/local/share/OpenMPI  -np 32 ./arc_proj02 -m 1 -n 10000 -w 100 -i ../DataGenerator/material.h5
