CudaDMA
====

The CudaDMA website can be found [here](http://lightsighter.github.io/CudaDMA/).

The CudaDMA library is a collection of DMA objects that support 
efficient movement of data between off-chip global memory and 
on-chip shared memory in CUDA kernels. CudaDMA objects support 
many different data transfer patterns including sequential, strided, 
gather, scatter, and halo patterns. CudaDMA objects provide both 
productivity and performance improvements in CUDA code:

* CudaDMA objects incorporate optimizations designed to fully utilize 
the underlying hardware and maximize memory bandwidth performance
* CudaDMA objects improve programmability by decoupling the declaration 
of a data transfer from its implementation. 

By handling the data-movement challenges on GPUs, CudaDMA makes it easier 
both to write CUDA code and achieve high performance.

CudaDMA is released under the [Apache License version 2.0](http://www.apache.org/licenses/LICENSE-2.0).
