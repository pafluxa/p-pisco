CC = nvcc
CXX = nvcc
CFLAGS = -Xcompiler -fopenmp -Xcompiler -fPIC -lm -O3 -DFORCE_INLINES -Xcompiler -g 
LDFLAGS = 
LOCAL_INC_DIRS = ./include
LOCAL_LIB_DIRS = ./lib 
EXT_INC_DIRS = $(PREFIX)/include
EXT_LIB_DIRS = $(PREFIX)/lib
SOURCES = ./src/*

libconvolve: 
	$(CXX) -shared $(CFLAGS) $(LDFLAGS) -I$(LOCAL_INC_DIRS) -I$(EXT_INC_DIRS) -L$(EXT_LIB_DIRS) -L$(LOCAL_LIB_DIRS) $(SOURCES) -o libconvolution.so
	mkdir -p $(PREFIX)/include/pisco/convolution
	mkdir -p $(PREFIX)/lib/
	mv libconvolution.so ./lib/
	cp ./lib/libconvolution.so $(PREFIX)/lib/
	cp include/*.h $(PREFIX)/include/pisco/convolution
