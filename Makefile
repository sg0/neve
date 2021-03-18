CXX = icpc
MPICXX = mpicxx

# use -xmic-avx512 instead of -xHost for Intel Xeon Phi platforms
OPTFLAGS = -O3 -xHost -DPRINT_DIST_STATS -DPRINT_EXTRA_NEDGES
# -DPRINT_EXTRA_NEDGES prints extra edges when -p <> is passed to 
#  add extra edges randomly on a generated graph
# use export ASAN_OPTIONS=verbosity=1 to check ASAN output
SNTFLAGS = -std=c++11 -fsanitize=address -O1 -fno-omit-frame-pointer
CXXFLAGS = -std=c++11 -g -I. $(OPTFLAGS)
CXXFLAGS_THREADS = -qopenmp -DUSE_SHARED_MEMORY -DGRAPH_FT_LOAD=4 #-DEDGE_AS_VERTEX_PAIR #-DENABLE_PREFETCH 
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
SRC_THREADS = main_threads.cpp
TARGET_THREADS = neve_threads

OBJS = $(OBJ_MPI) $(OBJ_THREADS)
TARGETS = $(TARGET_MPI) $(TARGET_THREADS)

all: $(TARGETS)
mpi: $(TARGET_MPI)
threads: $(TARGET_THREADS)

$(TARGET_MPI):  $(OBJ_MPI)
	$(LDAPP) $(MPICXX) -o $@ $+ $(LDFLAGS) $(CXXFLAGS) 

$(OBJ_MPI): $(SRC_MPI)
	$(MPICXX) $(INCLUDE) $(CXXFLAGS) -c $< -o $@

$(TARGET_THREADS):  $(OBJ_THREADS)
	$(LDAPP) $(CXX) $(CXXFLAGS_THREADS) -o $@ $+ $(LDFLAGS) $(CXXFLAGS) 

$(OBJ_THREADS): $(SRC_THREADS)
	$(CXX) $(INCLUDE) $(CXXFLAGS) $(CXXFLAGS_THREADS) -c $< -o $@

.PHONY: clean mpi threads

clean:
	rm -rf *~ *.dSYM nc.vg.* $(OBJS) $(TARGETS)
