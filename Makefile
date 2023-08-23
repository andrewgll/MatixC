CC=gcc
CFLAGS=-Iinclude -Wall -Wextra -O2 -pedantic -fstack-protector -fsanitize=address
CDEBUG=-g -Iinclude -Wall -Wextra -pedantic -O0 -fstack-protector -fsanitize=address
LDFLAGS=-lm

all: mylib

nnc:
	$(CC) $(CFLAGS) -c nnc.c -o build/nnc.o

tests: nnc
	$(CC) $(CDEBUG) tests.c build/nnc.o -o build/tests $(LDFLAGS) 
	valgrind --leak-check=full ./build/tests

clean:
	rm -rf build/*
