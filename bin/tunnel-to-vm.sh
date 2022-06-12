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

echo "zone=$ZONE, tpu=$TPU, project=$PROJECT, vm=$VM, task=$TASK"

gcloud config set account calcworks@gmail.com
gcloud config set project ${PROJECT}
gcloud config set compute/zone ${ZONE}

gcloud alpha compute tpus tpu-vm ssh ${VM} --zone ${ZONE} --project ${PROJECT} -- -v -NL 8080:localhost:8080
