#!/bin/bash

# Change the IP address (or machine name) with each restart.

# ADDR=54.213.82.156	#c5
ADDR=18.237.162.232		#p2
NAME=ubuntu
LHOST=localhost
SSHKEY=194-129-lstm/194.pem          # change if necessary to the name of your private key file

if $1; then
	for i in `seq 8888 8900`; do
	    FORWARDS[$((2*i))]="-L"
	    FORWARDS[$((2*i+1))]="$i:${LHOST}:$i"
	done
fi

if $2; then
	for i in `seq 6006 6007`; do
	    FORWARDS[$((2*i))]="-L"
	    FORWARDS[$((2*i+1))]="$i:${LHOST}:$i"
	done
fi

ssh -i ${SSHKEY} -X ${FORWARDS[@]} -l ${NAME} ${ADDR}