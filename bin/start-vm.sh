#!/usr/bin/env bash
# export PATH=~/courses/jax/git/30-days-of-jax-on-tpu/bin/:$PATH
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

echo -n "Choose a task (1 = start vm instance, 2 = list vm instances, 3 = delete vm instance): "
read TASK

export PROJECT=jax-tpu-getting-started
export VM=jtgs

echo "zone=$ZONE, tpu=$TPU, project=$PROJECT, vm=$VM, task=$TASK"

gcloud config set account calcworks@gmail.com
gcloud config set project ${PROJECT}
gcloud config set compute/zone ${ZONE}

if [[ TASK -eq 1 ]]; then
	echo -n "Do you want a preemptible instance (1 = Yes, 2 = No)? "  
	read PREEMPTIBLE

	if [[ PREEMPTIBLE -eq 1 ]]
	then
		export EXTRAS="--preemptible"
	else
		export EXTRAS=""
	fi	
	gcloud alpha compute tpus tpu-vm create ${VM} --zone ${ZONE} --accelerator-type ${TPU} --version tpu-vm-base $EXTRAS
elif [[ TASK -eq 2 ]]; then
  gcloud compute tpus tpu-vm list --zone ${ZONE}
else
  gcloud compute tpus tpu-vm delete ${VM} --zone ${ZONE} 
fi
