# Built OpenCL Square Program

CC = gcc
CFLAGS += -Ofast -march=native -mtune=native -std=c99 -Wall -D_DEFAULT_SOURCE -I.. -D CL_TARGET_OPENCL_VERSION=220 -Wextra -g
LDFLAGS += -lOpenCL

.PHONY: clean

square: square.c ../simple.o
	$(CC) $(CFLAGS) $^ -o $@ -lOpenCL

../simple.o: ../simple.c ../simple.h
	$(CC) -c $(CFLAGS) $< -o $@ -lOpenCL

clean:
	$(RM) ../simple.o square
