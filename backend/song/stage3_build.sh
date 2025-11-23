#!/bin/bash
# Build script for Stage 3 of the Song project

nvcc -std=c++14 -O3 SongStage3.cu main.cu -o stage3_final
