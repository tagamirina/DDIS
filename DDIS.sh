#!/bin/bash

for ((i=1 ; i<=1 ; i++))
do
if [ "$i" -ge 100 ]

then
	./DDIS -tmp ./MainData/tmp001.png -i ./MainData/img${i}.png -txt ./MainData/img${i}.txt -res ./IMG/output${i}.png -log ./TXT/output${i}.txt -RectDDIS ./IMG/heatmap${i}.png -v 1 -mode 1
else

	if [ "$i" -ge 10 ]

	then
		./DDIS -tmp ./MainData/tmp001.png -i ./MainData/img0${i}.png -txt ./MainData/img0${i}.txt -res ./IMG/output${i}.png -log ./TXT/output${i}.txt -RectDDIS ./IMG/heatmap${i}.png -v 1 -mode 1
	else
		./DDIS -tmp ./MainData/tmp001.png -i ./MainData/img00${i}.png -txt ./MainData/img00${i}.txt -res ./IMG/output${i}.png -log ./TXT/output${i}.txt -RectDDIS ./IMG/heatmap${i}.png -v 1 -mode 1
	fi
fi
done