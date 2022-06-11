#!/usr/bin/env bash

echo -n "Choose environment (1 = europe-west4-a/v3-8, 2 = us-central1-f/v2-8): "
read ENVIRONMENT

if [[ ENVIRONMENT -eq 1 ]]
then
	export ZONE=europe-west4-a
	export TPU=v3-8
fi

if [[ ENVIRONMENT -eq 2 ]]
then
	export ZONE=us-central1-f
	export TPU=v2-8
fi

export PROJECT=jax-tpu-getting-started
export VM=jtgs

echo "zone=$ZONE, tpu=$TPU, project=$PROJECT, vm=$VM"

gcloud config set account calcworks@gmail.com
gcloud config set project ${PROJECT}
gcloud config set compute/zone ${ZONE}

gcloud alpha compute tpus tpu-vm create ${VM} --zone ${ZONE} --accelerator-type ${TPU} --version tpu-vm-base
gcloud alpha compute tpus tpu-vm scp ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub ${VM}:~/.ssh/.
gcloud alpha compute tpus tpu-vm ssh ${VM} --zone ${ZONE} --project ${PROJECT}

# git config --global user.email "calcworks@gmail.com"
# git config --global user.name "Bob Flagg"
# git clone git@github.com:bobflagg/30-days-of-jax-on-tpu.git'
# cd /home/rflagg/30-days-of-jax-on-tpu/bin
# ./init.sh
# screen -S jupyter
# ./start-jupyter.sh
# gcloud alpha compute tpus tpu-vm ssh ${VM} --zone ${ZONE} --project ${PROJECT} -- -v -NL 8080:localhost:8080
# http://127.0.0.1:8080/?token=9d77aff9b04305bce60648ade73d281e9047e8320294b374

