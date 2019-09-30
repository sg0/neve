CXX = mpicxx
# use -xmic-avx512 instead of -xHost for Intel Xeon Phi platforms
OPTFLAGS = -O3 -xHost -DPRINT_DIST_STATS -DPRINT_EXTRA_NEDGES
# -DPRINT_EXTRA_NEDGES prints extra edges when -p <> is passed to 
#  add extra edges randomly on a generated graph
# use export ASAN_OPTIONS=verbosity=1 to check ASAN output
SNTFLAGS = -std=c++11 -fsanitize=address -O1 -fno-omit-frame-pointer
CXXFLAGS = -std=c++11 -g $(OPTFLAGS)

ENABLE_DUMPI_TRACE=0
ENABLE_SCOREP_TRACE=0

ifeq ($(ENABLE_DUMPI_TRACE),1)
	TRACERPATH = $(HOME)/builds/sst-dumpi/lib 
	LDFLAGS = -L$(TRACERPATH) -ldumpi
else ifeq ($(ENABLE_SCOREP_TRACE),1)
	SCOREP_INSTALL_PATH = /usr/common/software/scorep/6.0/intel
	CXXFLAGS += -DSCOREP_USER_ENABLE
	INCLUDE = -I$(SCOREP_INSTALL_PATH)/include -I$(SCOREP_INSTALL_PATH)/include/scorep -DSCOREP_USER_ENABLE
	LDAPP = $(SCOREP_INSTALL_PATH)/bin/scorep --user --nocompiler --noopenmp --nopomp --nocuda --noopenacc --noopencl --nomemory
endif

OBJ = main.o
TARGET = neve

all: $(TARGET)

%.o: %.cpp
	$(CXX) $(INCLUDE) $(CXXFLAGS) -c -o $@ $^

$(TARGET):  $(OBJ)
	$(LDAPP) $(CXX) -o $@ $^ 

.PHONY: clean

clean:
	rm -rf *~ $(OBJ) $(TARGET)
