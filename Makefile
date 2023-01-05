#
#    Makefile
#
#
#    Rina Tagami       Dec. 31, 2022.
#


CC       = g++
#CFLAGS    = -O2 -Wall -std=c++17
CFLAGS 	 = -O0 -fsanitize=address -fno-omit-frame-pointer -g -Wall -std=c++17

CVINC    = `pkg-config --cflags opencv`
CVLIB    = `pkg-config --libs opencv`
PATHS    = -I/usr/local/include -L/usr/local/lib -I/usr/local/include/opencv

clean :
	rm -f *.exe
	rm -f *.o
	rm -f *.a
	rm -f *.stackdump

ddis : DEMOrun.cpp computeDDIS.cpp run_TreeCANN.cpp DDIS_nnf_scan.cpp propagation_stage.cpp
	${CC} ${CFLAGS} -fopenmp DEMOrun.cpp computeDDIS.cpp run_TreeCANN.cpp DDIS_nnf_scan.cpp propagation_stage.cpp ${CVINC} ${CVLIB} -o DDIS
	#g++ -O2 -Wall -std=c++17 DEMOrun.cpp `pkg-config --cflags opencv` `pkg-config --libs opencv` -o DDIS