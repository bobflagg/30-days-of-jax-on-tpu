#!/usr/bin/env bash

jupyter notebook --generate-config
cat ./jupyter-config.txt >> /home/rflagg/.jupyter/jupyter_notebook_config.py
cd
export PYTHONPATH=~/30-days-of-jax-on-tpu/python
jupyter-notebook --no-browser --ip=0.0.0.0 --port=8080

