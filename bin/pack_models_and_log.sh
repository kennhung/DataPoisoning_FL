#!/bin/bash
tar jcvf $1/$(date +"%Y-%m-%d-%k-%M-%S").tar.bz2 ./logs/* 3*_models/ ./defense_results_3*.jpg