#!/bin/bash

nvcc -std=c++17 \
    main.cu \
    SongStage3.cu \
    SongStage1.cu \
    -o song_project
