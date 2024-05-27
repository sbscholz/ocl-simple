#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include <time.h>

#include <CL/cl.h>
#include "simple.h"

typedef struct {
  clarg_type arg_t;
  cl_mem dev_buf;
  double *double_host_buf;
  float *float_host_buf;
  int *int_host_buf;
  bool *bool_host_buf;
  int    num_elems;
  int    val;
  float valf;
  double vald;
} kernel_arg;

#define MAX_ARG 10


#define die(msg, ...) do {                      \
  (void) fprintf (stderr, msg, ## __VA_ARGS__); \
  (void) fprintf (stderr, "\n");                \
  exit (EXIT_FAILURE);                          \
} while (0)

/* global setup */

static bool verbose = false;
static cl_platform_id cpPlatform;     /* openCL platform.  */
static cl_device_id device_id;        /* Compute device id.  */
static cl_context context;            /* Compute context.  */
static cl_command_queue commands;     /* Compute command queue.  */
static cl_program program;            /* Compute program.  */
static int num_kernel_args;
static kernel_arg kernel_args[MAX_ARG];

static struct timespec start, stop;
static double kernel_time = 0.0;
static int num_kernel = 0;
static double h2d_time = 0.0;
static int num_h2d = 0;
static double d2h_time = 0.0;
static int num_d2h = 0;

#define CaseReturnString(x) case x: return #x;

const char *errToStr(cl_int err)
{
    switch (err) {
        CaseReturnString(CL_SUCCESS                        )
        CaseReturnString(CL_DEVICE_NOT_FOUND               )
        CaseReturnString(CL_DEVICE_NOT_AVAILABLE           )
        CaseReturnString(CL_COMPILER_NOT_AVAILABLE         )
        CaseReturnString(CL_MEM_OBJECT_ALLOCATION_FAILURE  )
        CaseReturnString(CL_OUT_OF_RESOURCES               )
        CaseReturnString(CL_OUT_OF_HOST_MEMORY             )
        CaseReturnString(CL_PROFILING_INFO_NOT_AVAILABLE   )
        CaseReturnString(CL_MEM_COPY_OVERLAP               )
        CaseReturnString(CL_IMAGE_FORMAT_MISMATCH          )
        CaseReturnString(CL_IMAGE_FORMAT_NOT_SUPPORTED     )
        CaseReturnString(CL_BUILD_PROGRAM_FAILURE          )
        CaseReturnString(CL_MAP_FAILURE                    )
        CaseReturnString(CL_MISALIGNED_SUB_BUFFER_OFFSET   )
        CaseReturnString(CL_COMPILE_PROGRAM_FAILURE        )
        CaseReturnString(CL_LINKER_NOT_AVAILABLE           )
        CaseReturnString(CL_LINK_PROGRAM_FAILURE           )
        CaseReturnString(CL_DEVICE_PARTITION_FAILED        )
        CaseReturnString(CL_KERNEL_ARG_INFO_NOT_AVAILABLE  )
        CaseReturnString(CL_INVALID_VALUE                  )
        CaseReturnString(CL_INVALID_DEVICE_TYPE            )
        CaseReturnString(CL_INVALID_PLATFORM               )
        CaseReturnString(CL_INVALID_DEVICE                 )
        CaseReturnString(CL_INVALID_CONTEXT                )
        CaseReturnString(CL_INVALID_QUEUE_PROPERTIES       )
        CaseReturnString(CL_INVALID_COMMAND_QUEUE          )
        CaseReturnString(CL_INVALID_HOST_PTR               )
        CaseReturnString(CL_INVALID_MEM_OBJECT             )
        CaseReturnString(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
        CaseReturnString(CL_INVALID_IMAGE_SIZE             )
        CaseReturnString(CL_INVALID_SAMPLER                )
        CaseReturnString(CL_INVALID_BINARY                 )
        CaseReturnString(CL_INVALID_BUILD_OPTIONS          )
        CaseReturnString(CL_INVALID_PROGRAM                )
        CaseReturnString(CL_INVALID_PROGRAM_EXECUTABLE     )
        CaseReturnString(CL_INVALID_KERNEL_NAME            )
        CaseReturnString(CL_INVALID_KERNEL_DEFINITION      )
        CaseReturnString(CL_INVALID_KERNEL                 )
        CaseReturnString(CL_INVALID_ARG_INDEX              )
        CaseReturnString(CL_INVALID_ARG_VALUE              )
        CaseReturnString(CL_INVALID_ARG_SIZE               )
        CaseReturnString(CL_INVALID_KERNEL_ARGS            )
        CaseReturnString(CL_INVALID_WORK_DIMENSION         )
        CaseReturnString(CL_INVALID_WORK_GROUP_SIZE        )
        CaseReturnString(CL_INVALID_WORK_ITEM_SIZE         )
        CaseReturnString(CL_INVALID_GLOBAL_OFFSET          )
        CaseReturnString(CL_INVALID_EVENT_WAIT_LIST        )
        CaseReturnString(CL_INVALID_EVENT                  )
        CaseReturnString(CL_INVALID_OPERATION              )
        CaseReturnString(CL_INVALID_GL_OBJECT              )
        CaseReturnString(CL_INVALID_BUFFER_SIZE            )
        CaseReturnString(CL_INVALID_MIP_LEVEL              )
        CaseReturnString(CL_INVALID_GLOBAL_WORK_SIZE       )
        CaseReturnString(CL_INVALID_PROPERTY               )
        CaseReturnString(CL_INVALID_IMAGE_DESCRIPTOR       )
        CaseReturnString(CL_INVALID_COMPILER_OPTIONS       )
        CaseReturnString(CL_INVALID_LINKER_OPTIONS         )
        CaseReturnString(CL_INVALID_DEVICE_PARTITION_COUNT )
        default: return "Unknown OpenCL error code";
    }
}

char *getMemStr( size_t n)
{
  static char buf[50];
  if (n>=1073741824) {
    snprintf( buf, 49, "%.2f GB", (float)n/1073741824.0);
  } else if (n>=1048576) {
    snprintf( buf, 49, "%.2f MB", (float)n/1048576.0);
  } else if (n>=1024) {
    snprintf( buf, 49, "%.2f KB", (float)n/1024.0);
  } else {
    snprintf( buf, 49, "%zu byte", n);
  }
  return buf;
}

char *getTimeStr( double time)
{
  static char buf[50];
  int min, sec;
  double msec;

  min = (int)time/60000;
  sec = (int)(time - (min*60000)) / 1000;
  msec = time - (min*60000) - (sec*1000);

  if (time >= 60000) {
    snprintf( buf, 49, "%d min %d sec %.1f msec", min, sec, msec);
  } else if (time >=1000) {
    snprintf( buf, 49, "%d sec %.1f msec", sec, msec);
  } else {
    snprintf( buf, 49, "%.1f msec", msec);
  }
  return buf;
}

char *readOpenCL( char *fname)
{
   FILE *f;
   size_t fsize;
   char *str;

   f = fopen(fname, "r");
   if (f==NULL)
      die ( "Error: file \"%s\" not found!", fname);

   fseek(f, 0, SEEK_END);
   fsize = ftell(f);
   fseek(f, 0, SEEK_SET);

   str = (char *)malloc(fsize + 1);
   if (str == NULL)
      die ("Error: failed to allocate memory for kernel of size %ld", fsize+1);
   size_t read_size;
   if ((read_size = fread(str, 1, fsize, f)) != fsize)
      die ("Error: read %zu bytes, expected %zu bytes", read_size, fsize);
   fclose(f);

   str[fsize] = 0;
   return str;
}

char *getPlatformName( cl_platform_id platform)
{
  size_t size;
  static char * res=NULL;

  CL_SAFE(clGetPlatformInfo( platform, CL_PLATFORM_NAME,0,NULL,&size));
  res = (char *)realloc( res, sizeof(char) * size + 1);
  CL_SAFE(clGetPlatformInfo( platform, CL_PLATFORM_NAME,size+1,res,NULL));
  return res;
}

cl_uint getDeviceMaxComputeUnits( cl_device_id device)
{
  cl_uint res;

  CL_SAFE(clGetDeviceInfo( device, CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&res,NULL));
  return res;
}

cl_ulong getMaxAlloc( cl_device_id device)
{
  cl_ulong res;

  CL_SAFE(clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,sizeof(cl_ulong),&res,NULL));
  return res;
}

cl_ulong getMemSize( cl_device_id device)
{
  cl_ulong res;

  CL_SAFE(clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(cl_ulong),&res,NULL));
  return res;
}

size_t getDeviceMaxWorkItems( cl_device_id device, int dim)
{
   size_t maxWI = 0;
   size_t max[3];

   if( dim >= 0 && dim < 3) {
      CL_SAFE(clGetDeviceInfo(device,
                            CL_DEVICE_MAX_WORK_ITEM_SIZES,
                            3*sizeof(size_t),
                            &max,
                            NULL));
      maxWI = max[dim];
   } else {
      die ("Error: maxWorkItems called with illegal parameter!");
   }

  return maxWI;
}

cl_int initDevice ( int devType)
{
  cl_int err = CL_SUCCESS;
  cl_uint num_platforms;
  cl_platform_id *cpPlatforms;
  cl_uint num_devices;
  cl_device_id *cpDevices = NULL;

  /* Connect to a compute device.  */
  err = clGetPlatformIDs (0, NULL, &num_platforms);
  if (CL_SUCCESS != err) {
    die ("Error: %s", errToStr(err));
  } else {
    cpPlatforms = (cl_platform_id *)malloc( sizeof( cl_platform_id)*num_platforms);
    err = clGetPlatformIDs(num_platforms, cpPlatforms, NULL);

    for(uint i=0; i<num_platforms; i++){
        if(verbose) {
          printf( "Platform[%d]: %s\n", i, getPlatformName( cpPlatforms[i]));
          err = clGetDeviceIDs(cpPlatforms[i], devType, 0, NULL, &num_devices);
          if (err == CL_SUCCESS ) {
            cpDevices = (cl_device_id *)realloc( cpDevices, sizeof( cl_device_id)*num_devices);
            err = clGetDeviceIDs(cpPlatforms[i], devType, num_devices, cpDevices, NULL);
            for(uint j=0; j<num_devices; j++){
              printf( "  Device[%d]: %d compute units\n", j, getDeviceMaxComputeUnits( cpDevices[j]));
              printf( "             %zux%zux%zu max local\n", getDeviceMaxWorkItems( cpDevices[j], 0),
                                                            getDeviceMaxWorkItems( cpDevices[j], 1),
                                                            getDeviceMaxWorkItems( cpDevices[j], 2));
              printf( "             %s global mem\n", getMemStr( getMemSize(cpDevices[j])));
            }
          } else {
            printf( "  no suitable device found\n");
          }
        }
        err = clGetDeviceIDs(cpPlatforms[i], devType, 1, &device_id, &num_devices);
        if (err == CL_SUCCESS ) {
           cpPlatform = cpPlatforms[i];
           if(verbose)
             printf( ">> Choosing platform %d\n", i);
           break;
        }
    }
    if (err != CL_SUCCESS) {
      die ("Error: %s", errToStr(err));
    } else {
      /* Create a compute context.  */
      context = clCreateContext (0, 1, &device_id, NULL, NULL, &err);
      if (!context || err != CL_SUCCESS) {
        die ("Error: %s", errToStr(err));
      } else {
        /* Create a command commands.  */
#ifdef CL_VERSION_2_0
        commands = clCreateCommandQueueWithProperties (context, device_id, 0, &err);
#else
        commands = clCreateCommandQueue (context, device_id, 0, &err);
#endif
        if (!commands || err != CL_SUCCESS) {
          die ("Error: %s", errToStr(err));
        }
      }
    }
  }

 return err;
}

cl_int initCPU ()
{
  return initDevice( CL_DEVICE_TYPE_CPU);
}

cl_int initGPU ()
{
  return initDevice( CL_DEVICE_TYPE_GPU);
}

cl_int initCPUVerbose ()
{
  verbose = true;
  return initDevice( CL_DEVICE_TYPE_CPU);
}

cl_int initGPUVerbose ()
{
  verbose = true;
  return initDevice( CL_DEVICE_TYPE_GPU);
}

size_t maxWorkItems( int dim)
{
   return getDeviceMaxWorkItems( device_id, dim);
}

cl_mem allocDev( size_t n)
{
   cl_int err = CL_SUCCESS;
   cl_mem mem;

   if (verbose)
     printf( "allocating %s on the device\n", getMemStr( n));
   mem = clCreateBuffer (context, CL_MEM_READ_WRITE, n, NULL, &err);
   if( err != CL_SUCCESS || mem == NULL)
      die ("Error: %s", errToStr(err));

   return mem;
}

#define H2D( tname, t)                                                          \
void host2dev ##tname ##Arr( t *a, cl_mem ad, size_t n)                         \
{                                                                               \
   clock_gettime( CLOCK_REALTIME, &start);                                      \
   if (verbose)                                                                 \
      printf( "transferring %s to device\n", getMemStr( sizeof (t) * n));       \
   CL_SAFE(clEnqueueWriteBuffer( commands, ad, CL_TRUE, 0,                      \
                               sizeof (t) * n,                                  \
                               a, 0, NULL, NULL));                              \
   clock_gettime( CLOCK_REALTIME, &stop);                                       \
   num_h2d++;                                                                   \
   h2d_time += (stop.tv_sec -start.tv_sec)*1000.0                               \
               + (stop.tv_nsec -start.tv_nsec)/1000000.0;                       \
}

H2D( Double, double)
H2D( Float, float)
H2D( Int, int)
H2D( Bool, bool)

#define D2H( tname, t)                                                         \
void dev2host ##tname ##Arr( cl_mem ad, t* a, size_t n)                        \
{                                                                              \
   clock_gettime( CLOCK_REALTIME, &start);                                     \
   if (verbose)                                                                \
      printf( "transferring %s to host\n", getMemStr( sizeof (t) * n));        \
   CL_SAFE(clEnqueueReadBuffer( commands, ad, CL_TRUE, 0,                      \
                              sizeof (t) * n,                                  \
                              a, 0, NULL, NULL));                              \
   clock_gettime( CLOCK_REALTIME, &stop);                                      \
   num_d2h++;                                                                  \
   d2h_time += (stop.tv_sec -start.tv_sec)*1000.0                              \
               + (stop.tv_nsec -start.tv_nsec)/1000000.0;                      \
}

D2H( Double, double)
D2H( Float, float)
D2H( Int, int)
D2H( Bool, bool)


cl_kernel createKernel( const char *kernel_source, char *kernel_name)
{
  cl_kernel kernel = NULL;
  cl_int err = CL_SUCCESS;

  /* Create the compute program from the source buffer.  */
  program = clCreateProgramWithSource (context, 1,
                                       (const char **) &kernel_source,
                                       NULL, &err);
  if (!program || err != CL_SUCCESS) {
    die ("Error: %s\n", errToStr(err));
  }

  /* Build the program executable.  */
  err = clBuildProgram (program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      size_t len;
      char buffer[2048];

      clGetProgramBuildInfo (program, device_id, CL_PROGRAM_BUILD_LOG,
                             sizeof (buffer), buffer, &len);
      die ("Error: Failed to build program executable!\n%s", buffer);
    }

  /* Create the compute kernel in the program.  */
  kernel = clCreateKernel (program, kernel_name, &err);
  if (!kernel || err != CL_SUCCESS) {
    die ("Error: Failed to create compute kernel!");
    kernel = NULL;
  }
  return kernel;
}

#define SETUPARG( tname, t)                                                      \
case tname ## Arr:                                                               \
   kernel_args[i].num_elems = va_arg(ap, int);                                   \
   kernel_args[i].t##_host_buf = va_arg(ap, t *);                                \
   kernel_args[i].dev_buf = allocDev ( sizeof (t) * kernel_args[i].num_elems);   \
   host2dev ## tname ## Arr ( kernel_args[i].t##_host_buf,                       \
                              kernel_args[i].dev_buf, kernel_args[i].num_elems); \
   CL_SAFE(clSetKernelArg (kernel, i, sizeof (cl_mem), &kernel_args[i].dev_buf);)\
break;

cl_kernel setupKernel( const char *kernel_source, char *kernel_name, int num_args, ...)
{
   cl_kernel kernel = NULL;
   va_list ap;
   int i;

   kernel = createKernel( kernel_source, kernel_name);
   num_kernel_args = num_args;
   va_start(ap, num_args);
   for(i=0; (i<num_args) && (kernel != NULL); i++) {
      kernel_args[i].arg_t =va_arg(ap, clarg_type);
      switch( kernel_args[i].arg_t) {
        SETUPARG( Double, double)
        SETUPARG( Float, float)
        SETUPARG( Int, int)
        SETUPARG( Bool, bool)
        case IntConst:
          kernel_args[i].val = va_arg(ap, unsigned int);
          CL_SAFE(clSetKernelArg (kernel, i, sizeof (unsigned int), &kernel_args[i].val));
          break;
        case FloatConst:
          /* Promoted because va_arg pushes to stack */
          kernel_args[i].valf = va_arg(ap, double);
          CL_SAFE(clSetKernelArg (kernel, i, sizeof (float), &kernel_args[i].valf));
          break;
        case DoubleConst:
          kernel_args[i].vald = va_arg(ap, double);
          CL_SAFE(clSetKernelArg (kernel, i, sizeof (double), &kernel_args[i].vald));
          break;
        default:
          die ("Error: illegal argument tag for executeKernel!");
      }
   }
   va_end(ap);

   return kernel;
}

cl_int launchKernel( cl_kernel kernel, int dim, size_t *global, size_t *local)
{
  cl_int err;
  if (verbose) {
    printf( "Trying to launch a kernel with global [ ");
    for(int i=0; i<dim; i++) {
      printf( "%zu ", global[i]);
    }
    printf( "] and local [ ");
    for(int i=0; i<dim; i++) {
      printf( "%zu ", local[i]);
    }
    printf( "]\n");
  }
  clock_gettime( CLOCK_REALTIME, &start);
  if (CL_SUCCESS
      != (err = clEnqueueNDRangeKernel (commands, kernel,
                                 dim, NULL, global, local, 0, NULL, NULL))) {
    if (!verbose) {
      printf( "Tried launching kernel with global [ ");
      for(int i=0; i<dim; i++) {
        printf( "%zu ", global[i]);
      }
      printf( "] and local [ ");
      for(int i=0; i<dim; i++) {
        printf( "%zu ", local[i]);
      }
      printf( "]\n");
    }
    die ("Error: %s", errToStr(err));
  }

  /* Wait for all commands to complete.  */
  CL_SAFE(clFinish (commands));
  clock_gettime( CLOCK_REALTIME, &stop);
  num_kernel++;
  kernel_time += (stop.tv_sec -start.tv_sec)*1000.0
                  + (stop.tv_nsec -start.tv_nsec)/1000000.0;

  return CL_SUCCESS;
}

#define FETCH( tname, t)                                     \
case tname ## Arr:                                           \
   dev2host ## tname ## Arr ( kernel_args[i].dev_buf,        \
                              kernel_args[i].t ## _host_buf, \
                              kernel_args[i].num_elems);     \
break;

cl_int runKernel( cl_kernel kernel, int dim, size_t *global, size_t *local)
{
  cl_int err = CL_SUCCESS;

  launchKernel( kernel, dim, global, local);

  for( int i=0; i< num_kernel_args; i++) {
      switch( kernel_args[i].arg_t) {
          FETCH( Double, double)
          FETCH( Float, float)
          FETCH( Int, int)
          FETCH( Bool, bool)
          case IntConst:
              /* do nothing */
              break;
          case FloatConst:
              /* do nothing */
              break;
          case DoubleConst:
              /* do nothing */
              break;
          default:
              die ("Error: illegal argument tag in runKernel!");
              kernel = NULL;
      }
  }

  return err;
}

void printKernelTime()
{
  printf( "total time spent in %d kernel executions: %s\n", num_kernel, getTimeStr( kernel_time));
}

void printTransferTimes()
{
  printf( "total time spent in %d host to device transfers : %s\n", num_h2d, getTimeStr( h2d_time));
  printf( "total time spent in %d device to host transfers : %s\n", num_d2h, getTimeStr( d2h_time));
}

cl_int freeDevice()
{
  for( int i=0; i< num_kernel_args; i++) {
    if( (kernel_args[i].arg_t == FloatArr)
         || (kernel_args[i].arg_t == DoubleArr))
      CL_SAFE(clReleaseMemObject (kernel_args[i].dev_buf));
  }
  CL_SAFE(clReleaseProgram (program));
  CL_SAFE(clReleaseCommandQueue (commands));
  CL_SAFE(clReleaseContext (context));

  return CL_SUCCESS;
}
