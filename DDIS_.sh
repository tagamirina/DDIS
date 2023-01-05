#!/bin/bash

for ((i=1 ; i<=105 ; i+=2))
do
	./DDIS -tmp ./data/${i}.jpg -i ./data/$((i+1)).jpg -txt ./data/${i}.txt -res ./outputIMG/output${i}.png -log ./outputTXT/output${i}.txt -RectDDIS ./outputIMG/heatmap${i}.png -v 1 -mode 0

done