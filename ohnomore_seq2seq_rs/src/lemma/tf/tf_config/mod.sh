#!/bin/sh

for f in $(find . -iname "*rs") 
	do
		echo $f | cut -d "/" -f 2 | cut -d "." -f 1 | sed -e 's/^/mod /'  >> mod.rs
	done	
