#! /bin/bash

SAMPLING=10
OPTIMIZATION=1566583817
TARGET=1566584416

rm -r ../exp_logs/$OPTIMIZATION/logs/$TARGET/latex
cp -r ./latex-src-summary ../exp_logs/$OPTIMIZATION/logs/$TARGET/latex
python ../parse.py --experiment_id $TARGET --optimization_id $OPTIMIZATION --downsampling $SAMPLING
cd ../exp_logs/$OPTIMIZATION/logs/$TARGET/latex
pdflatex --shell-escape main.tex
cd ./compiled/
for i in ./*.pdf
do
	inkscape $i --export-png=$i.png --export-dpi=200
done
xdg-open ../main.pdf
