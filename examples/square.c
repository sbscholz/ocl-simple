#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <CL/cl.h>
#include "simple.h"

#define DATA_SIZE 10240000

struct timespec start, stop;

void printTimeElapsed( char *text)
{
  double elapsed = (stop.tv_sec -start.tv_sec)*1000.0
                  + (double)(stop.tv_nsec -start.tv_nsec)/1000000.0;
  printf( "%s: %f msec\n", text, elapsed);
}

void timeDirectImplementation( int count, float* data, float* results)
{
  clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &start);
  for (int i = 0; i < count; i++)
    results[i] = data[i] * data[i];
  clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &stop);
  printTimeElapsed( "kernel equivalent on host");
}


int main (int argc, char * argv[])
{
  cl_int err;
  cl_kernel kernel;
  size_t global[1];
  size_t local[1];
  char *KernelSource = readOpenCL( "square.cl");

  if( argc <2) {
    local[0] = 32;
  } else {
    local[0] = atoi(argv[1]);
  }

  printf( "work group size: %d\n", (int)local[0]);

  clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &start);

  /* Create data for the run.  */
  float *data = NULL;                /* Original data set given to device.  */
  float *results = NULL;             /* Results returned from device.  */
  int correct;                       /* Number of correct results returned.  */

  int count = DATA_SIZE;
  global[0] = count;

  data = (float *) malloc (count * sizeof (float));
  results = (float *) malloc (count * sizeof (float));

  /* Fill the vector with random float values.  */
  for (int i = 0; i < count; i++)
    data[i] = rand () / (float) RAND_MAX;


  err = initGPUVerbose();
  
  if( err == CL_SUCCESS) {
    kernel = setupKernel( KernelSource, "square", 3, FloatArr, count, data,
                                                     FloatArr, count, results,
                                                     IntConst, count);

    runKernel( kernel, 1, global, local);
  
    clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &stop);

    printKernelTime();
    printTimeElapsed( "CPU time spent");

    /* Validate our results.  */
    correct = 0;
    for (int i = 0; i < count; i++)
      if (results[i] == data[i] * data[i])
        correct++;

    /* Print a brief summary detailing the results.  */
    printf ("Computed %d/%d %2.0f%% correct values\n", correct, count,
            (float)count/correct*100.f);

    err = clReleaseKernel (kernel);
    err = freeDevice();

    timeDirectImplementation( count, data, results);
    
  }


  return 0;
}


