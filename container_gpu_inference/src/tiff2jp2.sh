#!/bin/sh
infile=$1
outfile=$2
gdal_translate -of JP2OpenJPEG $1 $2
rm $1