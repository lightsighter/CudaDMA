![http://cudadma.googlecode.com/files/Logo.jpg](http://cudadma.googlecode.com/files/Logo.jpg)

The CudaDMA library is a collection of DMA objects that support efficient movement of data between off-chip global memory and on-chip shared memory in CUDA kernels.  CudaDMA objects support many different data transfer patterns including sequential, strided, gather, scatter, and halo patterns.  CudaDMA objects provide both productivity and performance improvements in CUDA code:

  * CudaDMA objects incorporate optimizations designed to fully utilize the underlying hardware and maximize memory bandwidth performance
  * CudaDMA objects improve programmability by decoupling the declaration of a data transfer from its implementation

By handling the data-movement challenges on GPUs, CudaDMA makes it easier both to write CUDA code and achieve high performance.

## IMPORTANT NOTE ##
CudaDMA will only be supported through the Kepler architecture.  NVIDIA is considering adopting CudaDMA as a supported CUDA library if there is sufficient interest.  Please contact us if you're interested in continuing to use CudaDMA in the future.

## CudaDMA API ##

The CudaDMA API currently supports two implementations: the original version that was written for Fermi, and a new version that has additional features for targeting Kepler but is also backwards compatible with Fermi.

[CudaDMA Version 1.0](http://code.google.com/p/cudadma/wiki/ProgrammingModel) (Fermi) contained in include/cudaDMA.h in the git repository
  * [Sequential](http://code.google.com/p/cudadma/wiki/CudaDMASequential)
  * [Strided](http://code.google.com/p/cudadma/wiki/CudaDMAStrided)
  * [Indirect](http://code.google.com/p/cudadma/wiki/CudaDMAIndirect)
  * [Halo](http://code.google.com/p/cudadma/wiki/CudaDMAHalo) (deprecated)

[CudaDMA Version 2.0](http://code.google.com/p/cudadma/wiki/CudaDMAv2) (Kepler+Fermi) contained in include/cudaDMAv2.h in the git repository
  * [Sequential](http://code.google.com/p/cudadma/wiki/CudaDMASequentialv2)
  * [Strided](http://code.google.com/p/cudadma/wiki/CudaDMAStridedv2)
  * [Indirect](http://code.google.com/p/cudadma/wiki/CudaDMAIndirectv2)


## Announcements ##
  * February 2013 - [CudaDMA Version 2.0](http://code.google.com/p/cudadma/wiki/CudaDMAv2) is now available.
  * January 2013 - Final versions of CudaDMA instances tuned for Kepler will be finished by early February.  The new instances will allow users to control the number of outstanding LDGs in flight on the K20 architecture.  They also contain optional knobs for controlling caching effects on loads and stores.  As before, all instances will support execution both with and without warp-specialization.
  * October 2012 - We've begun work on new CudaDMA instances tuned for the Kepler architecture.  The new instances will make use of the ldg intrinsic for supporting many more outstanding loads in flight by issuing them through the texture cache.  Work is ongoing but you can see observe our progress in the new cudaDMAK.h file (K for Kepler).
  * May 2012 - We've modified the implementation of the CudaDMAIndirect pattern so that different offsets can be supplied for each invocation of the `execute_dma` call for the same CudaDMA object.  Indexing for CudaDMAIndirect has also been made element-based instead of byte-based for easier use.
  * November 2011 - Based on feedback from users of CudaDMA at Supercomputing we have added additional support for using CudaDMA without warp specialization.  All CudaDMA instances are now parameterized by a boolean template parameter indicating whether a CudaDMA object is being used in conjunction with warp-specialization or not.  If warp specialization is not being employed, a simpler set of constructors are provided.
  * Slides from the CudaDMA technical talk at Supercomputing 2011 can be found [here](http://cudadma.googlecode.com/files/Supercomputing2011.pdf)

# Contributors #
[Mike Bauer](http://cs.stanford.edu/~mebauer)

[Henry Cook](http://www.eecs.berkeley.edu/~hcook/)

[Brucek Khailany](http://research.nvidia.com/users/brucek-khailany)