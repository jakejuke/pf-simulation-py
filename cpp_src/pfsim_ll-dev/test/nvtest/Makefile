CudaSources := $(wildcard *.cu)
CudaObjects := $(patsubst %.cu,%.o,$(CudaSources))
Targets := $(patsubst %.cu,%,$(CudaSources))
STD := -std=c++17
CPPFLAGS := $(STD) -I. -I../../
NVCCFLAGS :=
NVCC := nvcc
CXX := $(NVCC)
CC := $(NVCC)

.PHONY: all
all:    $(Targets)

$(CudaObjects): %.o: %.cu
	$(NVCC) -c $(CPPFLAGS) $(NVCCFLAGS) $<

$(Targets): %: %.o
	$(NVCC) -w -o $@ $(STD) $(NVCCFLAGS) $<

GCCSources := $(patsubst %,-x c++ %,$(CudaSources))
.PHONY: depend
depend:
	gcc-makedepend $(CPPFLAGS) -D__CUDACC__ $(GCCSources)

.PHONY: clean
clean:
	rm -f $(Targets) $(CudaObjects)

# DO NOT DELETE
