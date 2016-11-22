#!/bin/bash
for testfile in *.txt
do
	associatedFile="../"$testfile
	diff -q $testfile $associatedFile
done
echo "Compared all files."