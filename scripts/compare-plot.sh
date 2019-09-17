#! /bin/bash

SAMPLING=10
OPTIMIZATION=1566836061
targets=(
1566900027
1566900509
1566900994
1566903401
)

rm temp.txt
for i in "${targets[@]}"; do
	echo "$i" >> temp.txt
done
sort -o sorted.txt temp.txt
rm temp.txt
FOLDERNAME="compare"
while read line; do
	FOLDERNAME="$FOLDERNAME-$line"
done <sorted.txt
mkdir ../exp_logs/$OPTIMIZATION/logs/comparisons
mkdir ../exp_logs/$OPTIMIZATION/logs/comparisons/$FOLDERNAME
rm -r ../exp_logs/$OPTIMIZATION/logs/comparisons/$FOLDERNAME/latex/
cp -r ./latex-src-compare ../exp_logs/$OPTIMIZATION/logs/comparisons/$FOLDERNAME/latex/
counter=1
while read line; do
	if [ -d "../exp_logs/$OPTIMIZATION/logs/$line/latex/pgfplotsdata" ] 
	then
		echo "Warning: directory for $line already exists, just copying the data. Deleted the directory to force re-parsing." 
	else
		echo "Directory for $line does not exists. Parsing the data."
		mkdir ../exp_logs/$OPTIMIZATION/logs/$line/latex/
		mkdir ../exp_logs/$OPTIMIZATION/logs/$line/latex/pgfplotsdata/
		python ../parse.py --experiment_id $line --optimization_id $OPTIMIZATION --downsampling $SAMPLING
	fi
	cp -r ../exp_logs/$OPTIMIZATION/logs/$line/latex/pgfplotsdata/. ../exp_logs/$OPTIMIZATION/logs/comparisons/$FOLDERNAME/latex/pgfplotsdata/exp$counter/
	counter=$((counter+1))
done <sorted.txt
rm sorted.txt
cd ../exp_logs/$OPTIMIZATION/logs/comparisons/$FOLDERNAME/latex/
pdflatex --shell-escape main.tex
cd ./compiled/
for i in ./*.pdf
do
	inkscape $i --export-png=$i.png --export-dpi=200
done
xdg-open ../main.pdf
