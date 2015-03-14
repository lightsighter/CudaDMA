# CudaDMA Overview #

The goal of the CudaDMA library is to improve both programmability and performance when transferring data between global memory and shared memory in CUDA kernels.  To achieve these goals the library contains CudaDMA objects.  Each object describes a specific data transfer pattern.  Currently supported data transfer patterns include:

  * [Sequential](CudaDMASequential.md)
  * [Strided](CudaDMAStrided.md)
  * [Indirect](CudaDMAIndirect.md) (Gather/Scatter)
  * [Halo](CudaDMAHalo.md)

Each pattern implements the CudaDMA [API](Interface.md) and supports the CudaDMA [programming model](ProgrammingModel.md).

To get started using the CudaDMA library we recommend you begin [here](GettingStarted.md).

To optimize your CudaDMA code we describe multiple [buffering techniques](Buffering.md) that can be used in conjunction with the CudaDMA API.  We also describe some [restrictions](Restrictions.md) imposed by the CudaDMA API and the reasons for them.

To implement your own CudaDMA object we recommend reading our [best practices](BestPractices.md) guide.