# ocl-simple
This is a very simple wrapper for openCL.

Its main purpose is to help beginners getting started 
with openCL.

With it, a simple kernel can be set up and run by just 2 calls.
Here an excerpt from our example application:

```c
  err = initGPUVerbose();

  if( err == CL_SUCCESS) {
    kernel = setupKernel( KernelSource, "square", 3, FloatArr, count, data,
                                                     FloatArr, count, results,
                                                     IntConst, count);
    runKernel( kernel, 1, global, local);

    printKernelTime();
    printTransferTimes();

    err = clReleaseKernel (kernel);
    err = freeDevice();
  }
```

where `data` and `results` are pointers to host buffers and `count` is an integer variable on the host.

As soon as the interplay between host code and kernel code is understood,
slightly lower level wrappers can be used to control buffer allocations
and data transfers more explicitly while preserving some aid concerning
error messages, tracing and profiling. See simple.h for details.

