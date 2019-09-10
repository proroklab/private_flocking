#! /bin/bash

SAMPLING=10
OPTIMIZATION=1567621433

foldercounter=1
plotsperpage=10
pagecounter=10

rm temp.txt
for foldername in ../exp_logs/$OPTIMIZATION/logs/1*/ ; do
	echo "$foldername" | grep -o -E '/logs/[0-9]+' | grep -o -E '[0-9]+' >> temp.txt
	foldercounter=$((foldercounter+1))
	if [ "$foldercounter" = "$((plotsperpage+1))" ]; then
		echo "$pagecounter"
		FOLDERNAME="page-$pagecounter"
		mkdir ../exp_logs/$OPTIMIZATION/logs/evolution
		mkdir ../exp_logs/$OPTIMIZATION/logs/evolution/$FOLDERNAME
		rm -r ../exp_logs/$OPTIMIZATION/logs/evolution/$FOLDERNAME/latex/
		cp -r ./latex-src-evolution ../exp_logs/$OPTIMIZATION/logs/evolution/$FOLDERNAME/latex/
		counter=1
		while read line; do
			mkdir ../exp_logs/$OPTIMIZATION/logs/$line/latex/
			mkdir ../exp_logs/$OPTIMIZATION/logs/$line/latex/pgfplotsdata/
			python3.5 ../parse.py --experiment_id $line --optimization_id $OPTIMIZATION --downsampling $SAMPLING
			python3.5 ../json2tex.py $OPTIMIZATION $line
			cp -r ../exp_logs/$OPTIMIZATION/logs/$line/latex/pgfplotsdata/. ../exp_logs/$OPTIMIZATION/logs/evolution/$FOLDERNAME/latex/pgfplotsdata/exp$counter/
			counter=$((counter+1))
		done <temp.txt
		rm temp.txt
		cd ../exp_logs/$OPTIMIZATION/logs/evolution/$FOLDERNAME/latex/
		pdflatex --shell-escape main.tex
		open ./main.pdf
		cd ../../../../../../scripts
		pagecounter=$((pagecounter+1))
		foldercounter=1
	fi
done

flag=1
for foldername in ../exp_logs/$OPTIMIZATION/logs/evolution/* ; do
	if [ "$flag" = "1" ]; then
		cp $foldername/latex/main.pdf ../exp_logs/$OPTIMIZATION/logs/evolution/main.pdf
		flag=$((flag-1))
	else
	 	cp ../exp_logs/$OPTIMIZATION/logs/evolution/main.pdf ../exp_logs/$OPTIMIZATION/logs/evolution/temp.pdf
	 	rm ../exp_logs/$OPTIMIZATION/logs/evolution/main.pdf
	 	pdftk ../exp_logs/$OPTIMIZATION/logs/evolution/temp.pdf $foldername/latex/main.pdf cat output ../exp_logs/$OPTIMIZATION/logs/evolution/main.pdf
	fi
done
rm ../exp_logs/$OPTIMIZATION/logs/evolution/temp.pdf