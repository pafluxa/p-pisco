CC = gcc
CFLAGS = -std=c99 -fopenmp -lm -O3 -fPIC
LDFLAGS = -lsofa_c
LOCAL_INC_DIRS = ./include
LOCAL_LIB_DIRS = ./lib 
EXT_INC_DIRS = $(PREFIX)/include
EXT_LIB_DIRS = $(PREFIX)/lib
SOURCES = ./src/pointing_correction.c ./src/pointing_routines.c ./src/sph_trigo.c

libpointing: 
	$(CC) -shared $(CFLAGS) $(LDFLAGS) -I$(LOCAL_INC_DIRS) -I$(EXT_INC_DIRS) -L$(EXT_LIB_DIRS) -L$(LOCAL_LIB_DIRS) $(SOURCES) -o libpointing.so
	mkdir -p $(PREFIX)/include/pisco/pointing 
	mkdir -p $(PREFIX)/lib/
	mv libpointing.so ./lib/
	cp ./lib/libpointing.so $(PREFIX)/lib/
	cp include/*.h $(PREFIX)/include/pisco/pointing/
