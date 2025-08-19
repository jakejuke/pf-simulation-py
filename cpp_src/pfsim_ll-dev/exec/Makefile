CXX:=g++
RM:=rm

LIBFLAGS:= -I./../ -fopenmp
MACROS:=
STD:=-std=c++0x
CFLAGS:=-Wall -g -O3 $(STD) $(LIBFLAGS) $(MACROS) -mavx

Targets:=sim2d simpp2d simiso2d simisopp2d simiso3d sim3d sim3di simpp3d simisopp3d simanisoMob3d simSingleGrainWets3d

.PHONY: all clean 

all: $(Targets)

clean: 
	$(RM) $(Targets) generate_misorientation generate_particlefield[23]d

generate_misorientation: generate_misorientation.cpp
	$(CXX) $(CFLAGS) -DNOF_GRAINS=39 -o $@ $^

misorientation: generate_misorientation
	./$^

generate_particlefield%: generate_particlefield%.cpp
	$(CXX) $(CFLAGS) -o $@ $^

particlefield%: generate_particlefield%
	./$^

%2d: %2d.cpp
	$(CXX) $(CFLAGS) -o $@ $^
	
%3d: %3d.cpp
	$(CXX) $(CFLAGS) -o $@ $^

%3di: %3di.cpp
	$(CXX) $(CFLAGS) -o $@ $^
