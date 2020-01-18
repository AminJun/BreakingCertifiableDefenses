#!/bin/bash 

function git_sync(){
	while true ; do 
		if [ $# -gt 0 ] ; then 
			file="$1"
			time="$(date)"
			git add "${file}"
			git commit -m "Automatic Sync of ${time} on file ${file}"
			git push
		else 
			git pull 
		fi 
		sleep 1
	done
}
