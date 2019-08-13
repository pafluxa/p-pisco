CC = gcc
CFLAGS = -std=c99 -fopenmp -lm -O3 -fPIC
LDFLAGS = -llapack -lblas
LOCAL_INC_DIRS = ./include
LOCAL_LIB_DIRS = ./lib 
EXT_INC_DIRS = $(PREFIX)/include
EXT_LIB_DIRS = $(PREFIX)/lib
SOURCES = ./src/*.c

libmapping: 
	$(CC) -shared $(CFLAGS) $(LDFLAGS) -I$(LOCAL_INC_DIRS) -I$(EXT_INC_DIRS) -L$(EXT_LIB_DIRS) -L$(LOCAL_LIB_DIRS) $(SOURCES) -o libmapping.so
	mkdir -p $(PREFIX)/include/pisco/mapping
	mkdir -p $(PREFIX)/lib/
	mv libmapping.so ./lib/
	cp ./lib/libmapping.so $(PREFIX)/lib/
	cp include/*.h $(PREFIX)/include/pisco/mapping
