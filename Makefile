CC=gcc
INCLUDE=-Iinclude
COMMON_FLAGS=$(INCLUDE) -Wall -Wextra -pedantic -fstack-protector 

# Release flags
CFLAGS=-O2 $(COMMON_FLAGS)
LDFLAGS=-lm

# Debug flags
CDEBUGFLAGS=-g -O0 $(COMMON_FLAGS)
LDEBUGFLAGS=-lm

OBJ_DIR=build
OBJS=$(OBJ_DIR)/nnc.o

.PHONY: all debug clean tests

all: CFLAGS+=$(CFLAGS)
all: LDFLAGS+=$(LDFLAGS)
all: mylib

debug: CFLAGS+=$(CDEBUGFLAGS)
debug: LDFLAGS+=$(LDEBUGFLAGS)
debug: mylib

nnc:
	$(CC) $(CFLAGS) -c nnc.c -o $(OBJ_DIR)/nnc.o

tests: nnc
	$(CC) $(CFLAGS) tests.c $(OBJS) -o $(OBJ_DIR)/tests $(LDFLAGS)
	valgrind --leak-check=full ./$(OBJ_DIR)/tests

clean:
	rm -rf $(OBJ_DIR)/*
