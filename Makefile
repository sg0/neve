ENABLE_OMP_OFFLOAD=1
ENABLE_PINNED=1
CXX = g++
ifeq ($(ENABLE_OMP_OFFLOAD),1)
CXX = clang++
endif
MPICXX = mpicxx
NVCC = nvcc
SM=70

# use -xmic-avx512 instead of -xHost for Intel Xeon Phi platforms
OPTFLAGS = -O3 -DPRINT_DIST_STATS -DPRINT_EXTRA_NEDGES -std=c++11
# -DPRINT_EXTRA_NEDGES prints extra edges when -p <> is passed to 
#  add extra edges randomly on a generated graph
# use export ASAN_OPTIONS=verbosity=1 to check ASAN output
SNTFLAGS = -std=c++11 -fsanitize=address -O1 -fno-omit-frame-pointer
#CXXFLAGS = -std=c++11 -g -I. $(OPTFLAGS)
CXXFLAGS = -g -I. $(OPTFLAGS)
#CXXFLAGS = -I. $(OPTFLAGS)
#CXXFLAGS_THREADS = -fopenmp -DUSE_SHARED_MEMORY -DGRAPH_FT_LOAD=4 -DNTIMES=20 #-DEDGE_AS_VERTEX_PAIR #-DENABLE_PREFETCH 
CXXFLAGS_THREADS = -fopenmp
ifeq ($(ENABLE_OMP_OFFLOAD),1)
CXXFLAGS_THREADS += -fopenmp-targets=nvptx64 -Xopenmp-target=nvptx64 -march=sm_${SM} --gcc-toolchain=/share/apps/gcc/9.1.0 -DUSE_OMP_ACCELERATOR
endif
CXXFLAGS_THREADS += -DUSE_SHARED_MEMORY -DGRAPH_FT_LOAD=4 -DNTIMES=20 #-I/usr/lib/gcc/x86_64-redhat-linux/4.8.5/include/

CUFLAGS = -O3 -Xptxas -O3 --std=c++14 --gpu-architecture=compute_${SM} --gpu-code=sm_${SM},compute_${SM} \
-Xcompiler -O3 -Xcompiler -fopenmp -DUSE_SHARED_MEMORY -DUSE_CUDA -DGRAPH_FT_LOAD=2 -DNTIMES=20 
ifeq ($(ENABLE_PINNED),1)
CUFLAGS += -DUSE_PINNED_HOST
endif

CXXFLAGS_MPI = 
ENABLE_DUMPI_TRACE=0
ENABLE_SCOREP_TRACE=0
ifeq ($(ENABLE_DUMPI_TRACE),1)
	TRACERPATH = $(HOME)/builds/sst-dumpi/lib 
	LDFLAGS = -L$(TRACERPATH) -ldumpi
else ifeq ($(ENABLE_SCOREP_TRACE),1)
	SCOREP_INSTALL_PATH = /usr/common/software/scorep/6.0/intel
	INCLUDE = -I$(SCOREP_INSTALL_PATH)/include -I$(SCOREP_INSTALL_PATH)/include/scorep -DSCOREP_USER_ENABLE
	LDAPP = $(SCOREP_INSTALL_PATH)/bin/scorep --user --nocompiler --noopenmp --nopomp --nocuda --noopenacc --noopencl --nomemory
endif

ENABLE_SSTMACRO=0
ifeq ($(ENABLE_SSTMACRO),1)
    SSTPATH = $(HOME)/builds/sst-macro
    CXX = $(SSTPATH)/bin/sst++
    CXXFLAGS += -fPIC -DSSTMAC -I$(SSTPATH)/include
    LDFLAGS = -Wl,-rpath,$(SSTPATH)/lib -L$(SSTPATH)/lib
endif

# https://software.llnl.gov/Caliper/services.html
ENABLE_LLNL_CALIPER=0
ifeq ($(ENABLE_LLNL_CALIPER), 1)
CALI_PATH = $(HOME)/builds/caliper
CXXFLAGS += -DLLNL_CALIPER_ENABLE -I$(CALI_PATH)/include
LDFLAGS = -Wl,-rpath,$(CALI_PATH)/lib -L$(CALI_PATH)/lib -lcaliper
endif

OBJ_MPI = main.o
SRC_MPI = main.cpp
TARGET_MPI = neve_mpi 
OBJ_THREADS = main_threads.o
OBJ_CUDA = main_cuda.o
OBJ_CUDA_BATCH = main_cuda_batch.o graph_gpu.o graph_cuda.o
OBJ_CUDA_SORT = main_cuda_sort.o graph_gpu.o graph_cuda.o
SRC_THREADS = main_threads.cpp
SRC_CUDA = main_cuda.cpp
TARGET_THREADS = neve_threads
TARGET_CUDA = neve_cuda
TARGET_CUDA_BATCH = neve_cuda_batch
TARGET_CUDA_SORT = neve_cuda_sort

OBJS = $(OBJ_MPI) $(OBJ_THREADS) ${OBJ_CUDA} ${OBJ_CUDA_BATCH} ${OBJ_CUDA_SORT}
TARGETS = $(TARGET_MPI) $(TARGET_THREADS) ${TARGET_CUDA} ${TARGET_CUDA_BATCH} ${TARGET_CUDA_SORT}

all: $(TARGETS)
mpi: $(TARGET_MPI)
threads: $(TARGET_THREADS)
cuda: ${TARGET_CUDA}
cuda_batch: ${TARGET_CUDA_BATCH}
cuda_sort: ${TARGET_CUDA_SORT}

$(TARGET_MPI):  $(OBJ_MPI)
	$(LDAPP) $(MPICXX) -o $@ $+ $(LDFLAGS) $(CXXFLAGS) 

$(OBJ_MPI): $(SRC_MPI)
	$(MPICXX) $(INCLUDE) $(CXXFLAGS) -c $< -o $@

$(TARGET_THREADS):  $(OBJ_THREADS)
	$(LDAPP) $(CXX) $(CXXFLAGS_THREADS) -o $@ $+ $(LDFLAGS) $(CXXFLAGS) 

${TARGET_CUDA}:  $(OBJ_CUDA)
	$(NVCC) $(CUFLAGS) -o $@ $^

${TARGET_CUDA_BATCH}: main_cuda_batch.o graph_cuda.o graph_gpu.o
	$(NVCC) $(CUFLAGS) -o $@ $^

${TARGET_CUDA_SORT}:  main_cuda_sort.o graph_cuda.o graph_gpu.o
	 $(NVCC) $(CUFLAGS) -o $@ $^

$(OBJ_THREADS): $(SRC_THREADS)
	$(CXX) $(INCLUDE) $(CXXFLAGS) $(CXXFLAGS_THREADS) -c $< -o $@

${OBJ_CUDA}: ${SRC_CUDA} graph.cuh
	$(NVCC) $(INCLUDE) -x cu $(CUFLAGS) -dc $< -o $@

main_cuda_batch.o: main_cuda_batch.cpp graph.hpp graph_gpu.hpp
	$(NVCC) $(INCLUDE) -x cu $(CUFLAGS) -dc $< -o $@

main_cuda_sort.o: main_cuda_sort.cpp graph.hpp graph_gpu.hpp
	$(NVCC) $(INCLUDE) -x cu $(CUFLAGS) -dc $< -o $@

graph_cuda.o:graph_cuda.cu graph_cuda.hpp 
	$(NVCC) $(CUFLAGS) -o $@ -c $<

graph_gpu.o: graph_gpu.cpp graph_gpu.hpp
	$(NVCC) $(INCLUDE) -x cu $(CUFLAGS) -dc $< -o $@

.PHONY: clean mpi threads cuda

clean:
	rm -rf *~ *.dSYM nc.vg.* $(OBJS) $(TARGETS)
