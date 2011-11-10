#
#  Copyright 2010 NVIDIA Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


DEPS = params.h params_common.h params_directed.h ../../../include/cudaDMA.h saxpy_cudaDMA.cu saxpy_cudaDMA_kernel.cu

all: saxpy_cudaDMA

saxpy_cudaDMA: $(DEPS)
	nvcc -o saxpy_cudaDMA -O2 -arch=compute_20 -code=sm_20 saxpy_cudaDMA.cu

clean:
	rm -f *.o saxpy_cudaDMA
