#include <cuda_runtime.h>
#include <stdio.h>

#define N 9

__global__ void kernel(float *a, float *b, float *c, int batch_size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < batch_size)
		c[i] = a[i] + b[i];
}

int main()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	float a[N] = {0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.5};
	float b[N] = {0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.5};
	float c[N];

	cudaStream_t *streams = new cudaStream_t[deviceCount];
/*
	float **d_a = new float*[deviceCount];
	float **d_b = new float*[deviceCount];
	float **d_c = new float*[deviceCount];
*/

	// make device data ptr for each GPU
	float *d_a[deviceCount];
	float *d_b[deviceCount];
	float *d_c[deviceCount];

	int batch_size = (N + deviceCount - 1)/ deviceCount;

	// create stream for each GPU and malloc the device memory
	for (int i = 0; i < deviceCount; ++i)
	{
		cudaSetDevice(i);
		cudaStreamCreate(&streams[i]);
		cudaMalloc(&d_a[i], batch_size * sizeof(float));
		cudaMalloc(&d_b[i], batch_size * sizeof(float));
		cudaMalloc(&d_c[i], batch_size * sizeof(float));
	}

	for (int i = 0; i < deviceCount; ++i)
	{
		cudaSetDevice(i);
		cudaMemcpyAsync(d_a[i], &a[i * batch_size], batch_size * sizeof(float),
						cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(d_b[i], &b[i * batch_size], batch_size * sizeof(float),
						cudaMemcpyHostToDevice, streams[i]);
	}

	dim3 grid = 1;
	dim3 block = batch_size;
	for (int i = 0; i < deviceCount; ++i)
	{
		cudaSetDevice(i);
		kernel<<<grid, block, 0, streams[i]>>>(d_a[i], d_b[i], d_c[i],
											   batch_size);
	}

	for (int i = 0; i < deviceCount; ++i)
	{
		cudaSetDevice(i);
		cudaMemcpyAsync(&c[i * batch_size], d_c[i], batch_size * sizeof(float),
						cudaMemcpyDeviceToHost, streams[i]);
	}

	for (int i = 0; i < deviceCount; ++i)
	{
		cudaSetDevice(i);
		cudaStreamSynchronize(streams[i]);
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess)
			printf("stream-%d error: %s\n", i, cudaGetErrorString(error));
	}
	
	for (int i = 0; i < N; ++i)
		printf("c[%d]=%f\n", i, c[i]);

	for (int i = 0; i < deviceCount; ++i)
	{
		cudaSetDevice(i);
		cudaFree(d_a[i]);
		cudaFree(d_b[i]);
		cudaFree(d_c[i]);
		cudaStreamDestroy(streams[i]);
	}

	delete[] streams;

	return 0;
}

