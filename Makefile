CXX = CC
# use -xmic-avx512 instead of -xHost for Intel Xeon Phi platforms
OPTFLAGS = -O3 -xHost -qopenmp -DPRINT_DIST_STATS -DPRINT_EXTRA_NEDGES
# -DPRINT_EXTRA_NEDGES prints extra edges when -p <> is passed to 
#  add extra edges randomly on a generated graph
# use export ASAN_OPTIONS=verbosity=1 to check ASAN output
SNTFLAGS = -std=c++11 -fopenmp -fsanitize=address -O1 -fno-omit-frame-pointer
CXXFLAGS = -std=c++11 -g $(OPTFLAGS)

ENABLE_TRACE=0
ifeq ($(ENABLE_TRACE),1)
TRACERPATH = $(HOME)/builds/sst-dumpi/lib 
LDFLAGS = -L$(TRACERPATH) -ldumpi
endif

OBJ = main.o
TARGET = neve

all: $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $^

$(TARGET):  $(OBJ)
	$(CXX) $^ $(OPTFLAGS) -o $@ $(LDFLAGS)

.PHONY: clean

clean:
	rm -rf *~ $(OBJ) $(TARGET)
