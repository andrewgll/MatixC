CC=gcc
INCLUDE=-Iinclude
UNITY_DIR=Unity
UNITY_SRC_DIR=$(UNITY_DIR)/src
UNITY_FLAGS=-I$(UNITY_SRC_DIR)
UNITY_REPO=https://github.com/ThrowTheSwitch/Unity.git
COMMON_FLAGS=$(INCLUDE) $(UNITY_FLAGS) -Wall -Wextra -pedantic -fstack-protector 

DOUBLE_PRECISION_FLAGS=-DUNITY_INCLUDE_DOUBLE -DUSE_DOUBLE_PRECISION

# Release flags
CFLAGS=-O2 $(COMMON_FLAGS)
LDFLAGS=-lm

# Debug flags
CDEBUGFLAGS=-g -O0 $(COMMON_FLAGS)
LDEBUGFLAGS=-lm

OBJ_DIR=build
OBJS=$(OBJ_DIR)/nnc.o
APP_EXECUTABLE=$(OBJ_DIR)/nnc.o
UNITY_TEST_EXECUTABLE=$(OBJ_DIR)/unity_test

.PHONY: all debug clean tests run unity_tests

all: CFLAGS+=$(CFLAGS)
all: LDFLAGS+=$(LDFLAGS)
all: nnc

debug: CFLAGS+=$(CDEBUGFLAGS)
debug: LDFLAGS+=$(LDEBUGFLAGS)
debug: debug

nnc:
	$(CC) $(CFLAGS) -c nnc.c -o $(OBJ_DIR)/nnc.o
nnc_debug:
	$(CC) $(CDEBUGFLAGS) -c nnc.c -o $(OBJ_DIR)/nnc.o

example: nnc_debug
	$(CC) $(CDEBUGFLAGS) example.c $(OBJS) -o $(OBJ_DIR)/example $(LDFLAGS)

tests: clone_unity nnc_debug
	$(CC) $(CDEBUGFLAGS) tests.c $(OBJS) $(UNITY_SRC_DIR)/unity.c -o $(UNITY_TEST_EXECUTABLE) $(LDEBUGFLAGS)
	valgrind --leak-check=full --track-origins=yes $(UNITY_TEST_EXECUTABLE)

clone_unity:
	if [ ! -d $(UNITY_DIR) ]; then git clone $(UNITY_REPO) $(UNITY_DIR); fi

run: example
	$(OBJ_DIR)/example

clean:
	rm -rf $(OBJ_DIR)/*
