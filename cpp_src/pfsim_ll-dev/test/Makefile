CC:=g++
RM:=rm

LIBFLAGS:= -I./../
MACROS:= 
STD:= -std=c++0x 
CFLAGS:= -Wall -O3 -g $(STD) $(LIBFLAGS) $(MACROS)

.PHONY: clean 


clean: 
	$(RM) a.out 

plotball: plotball.cpp
	g++ -Wall  -I./../ -fopenmp -o plotball plotball.cpp `root-config --cflags --glibs`

%: %.cpp
	$(CC) $(CFLAGS) -o $@ $^
