#!/bin/bash 

if [ $# -gt 0 ] ; then 
	name_of_file="$1"
	par="results/${name_of_file}"
	mkdir -p "${par}"
	attack="${par}/attack_cert.txt"
	base="${par}/base_cert.txt"
	rm "${attack}"
	rm "${base}"

	for i in `seq 0 100 10000`; do 
		cat "${name_of_file}" | grep "^${i}[[:space:]]" | tail -n 1 | awk '{print $11}' >> "${attack}"
		cat "${name_of_file}" | grep "^${i}[[:space:]]" | tail -n 1 | awk '{print $6}'  >> "${base}"
	done
fi
