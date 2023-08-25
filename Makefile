CC=clang
INCLUDE=-Iinclude
UNITY_DIR=Unity
UNITY_SRC_DIR=$(UNITY_DIR)/src
UNITY_FLAGS=-I$(UNITY_SRC_DIR)
UNITY_REPO=https://github.com/ThrowTheSwitch/Unity.git
COMMON_FLAGS=$(INCLUDE) $(UNITY_FLAGS) -Wall -Wextra -pedantic -fstack-protector 

DOUBLE_PRECISION_FLAGS=-DUNITY_INCLUDE_DOUBLE -DUSE_DOUBLE_PRECISION
LDFLAGS=-lm

OBJ_DIR=build
OBJS=$(OBJ_DIR)/mx.o
UNITY_TEST_EXECUTABLE=$(OBJ_DIR)/unity_test

ANALYSIS_CHECKERS=-enable-checker core -enable-checker alpha -enable-checker unix -enable-checker cplusplus


# Release flags
CFLAGS=-O2 $(COMMON_FLAGS)

# Debug flags
CDEBUGFLAGS=-g -O0 $(COMMON_FLAGS)

# Static analysis output directory
ANALYSIS_OUTPUT_DIR=analysis_output

.PHONY: all debug clean tests run unity_tests analyze

all: mx

debug: mx_debug

mx:
	$(CC) $(CFLAGS) -c mx.c -o $(OBJ_DIR)/mx.o

mx_debug:
	$(CC) $(CDEBUGFLAGS) -c mx.c -o $(OBJ_DIR)/mx.o

example: mx_debug
	$(CC) $(CDEBUGFLAGS) example2.c $(OBJS) -o $(OBJ_DIR)/example $(LDFLAGS)

tests: clone_unity mx_debug
	$(CC) $(CDEBUGFLAGS) tests.c $(OBJS) $(UNITY_SRC_DIR)/unity.c -o $(UNITY_TEST_EXECUTABLE) $(LDFLAGS)
	valgrind --leak-check=full --track-origins=yes $(UNITY_TEST_EXECUTABLE)

clone_unity:
	if [ ! -d $(UNITY_DIR) ]; then git clone $(UNITY_REPO) $(UNITY_DIR); fi

run: example
	$(OBJ_DIR)/example

clean:
	rm -rf $(OBJ_DIR)/* $(ANALYSIS_OUTPUT_DIR)

# Static analysis target
analyze:
	scan-build $(ANALYSIS_CHECKERS) -o $(ANALYSIS_OUTPUT_DIR) make all
