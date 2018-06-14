#!/bin/bash

animalName=$1
let maxNbr=$2
dir="Images/"$animalName
mkdir -p $dir
let i=1

while read p; do
	if [ "$i" -gt "$maxNbr" ]; then
		break
	fi
	fname=$dir"/"$animalName$i
	p=${p%$'\r'}
	curl $p --max-time 10 > $fname
	ftype=$(file $fname)
	if [[ $ftype = *JPEG* ]]; then
		let i+=1
	else
		rm $fname
	fi
done < $animalName"URLs"
