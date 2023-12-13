#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <nvml.h>

#define GET_CURRENT_MICRO	std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()

#define cuda_print(X) \
{\
	cudaError_t err = X;\
	printf("cuda_print %s status=%d, %s\n",\
	#X, err, cudaGetErrorName(err));\
}

#define N 33

float a[N][N];
float b[N][N];
float c[N][N];

float (*d_a)[N];
float (*d_b)[N];
float (*d_c)[N];

uint64_t begin, end;
cudaError_t status;
float epsilon = 0.0001;

int block_num = 1;

dim3 thread_per_block(N, N); // this is grid
dim3 block(N / thread_per_block.x, N / thread_per_block.y); // block_num

__global__ void sum_matrix_parallel(float a[N][N], float b[N][N], float c[N][N])
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < N && j < N) {
		c[i][j] = a[i][j] + b[i][j];
	}
}

__global__ void sum_matrix_serial(float a[N][N], float b[N][N], float c[N][N])
{
	for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++)
		c[i][j] = a[i][j] + b[i][j];
}

void print_parameters()
{
	int devicecount;
	cudaGetDeviceCount(&devicecount);

	for (int i = 0; i < devicecount; ++i)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);

		printf("Device %d: %s\n", i, deviceProp.name);
		printf("compute capability Version: %d.%d\n",
			   deviceProp.major, deviceProp.minor);
		printf("Max Threads per block: %d\n",
			   deviceProp.maxThreadsPerBlock);
		printf("Max Threads per Multiprocessor: %d\n",
			   deviceProp.maxThreadsPerMultiProcessor);

		printf("\n");
	}
}

void print_utilization()
{
	nvmlDevice_t device;
	nvmlUtilization_t utilization;

	begin = GET_CURRENT_MICRO;
	nvmlInit();
	end = GET_CURRENT_MICRO;
	printf("nvmlInit cost = %llu\n", end - begin);

	begin = GET_CURRENT_MICRO;
	nvmlDeviceGetHandleByIndex(0, &device);
	end = GET_CURRENT_MICRO;
	printf("nvmlDeviceGetHandleByIndex cost = %llu\n", end - begin);

	begin = GET_CURRENT_MICRO;
	nvmlDeviceGetUtilizationRates(device, &utilization);
	end = GET_CURRENT_MICRO;
	printf("nvmlDeviceGetUtilizationRates cost = %llu\n", end - begin);

	printf("GPU utilization: %d%%\n", utilization.gpu);
	printf("Memory utilization: %d%%\n", utilization.memory);

	nvmlShutdown();
	end = GET_CURRENT_MICRO;
	printf("nvmlShutdown cost = %llu\n", end - begin);
}

int init_and_malloc()
{
	begin = GET_CURRENT_MICRO; 
	status = cudaMalloc((void**)&d_a, N * N * sizeof(float));
	if (status != cudaSuccess)
		return -1;

	status = cudaMalloc((void**)&d_b, N * N * sizeof(float));
	if (status != cudaSuccess)
		return -1;

	status = cudaMalloc((void**)&d_c, N * N * sizeof(float));
	if (status != cudaSuccess)
		return -1;
	end = GET_CURRENT_MICRO;
	printf("cudaMalloc cost = %llu\n", end - begin);

	for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++)
	{
		a[i][j] = i + 0.2;
		b[i][j] = j - 0.1;
		c[i][j] = 0;
	}

	begin = GET_CURRENT_MICRO;
	status = cudaMemcpy(d_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		return -1;

	status = cudaMemcpy(d_b, b, N * N * sizeof(float), cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		return -1;
	end = GET_CURRENT_MICRO;
	printf("cudaMemcpy cost = %llu\n", end - begin);

	return 0;
}

void test_parallel()
{
	sum_matrix_parallel<<<block_num, thread_per_block>>>(d_a, d_b, d_c);
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		cuda_print(status);
		return;
	}

	cudaDeviceSynchronize();
}

void test_serial()
{
	sum_matrix_serial<<<block_num, 1>>>(d_a, d_b, d_c);
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		cuda_print(status);
		return;
	}

	cudaDeviceSynchronize();
}

void deinit_and_free()
{
	begin = GET_CURRENT_MICRO;
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	end = GET_CURRENT_MICRO;
	printf("cudaFree cost = %llu\n", end - begin);
}

void copy_and_check()
{
	status = cudaMemcpy(c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
	{
		cuda_print(status);
		return;
	}

	for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++)
	{
		if (fabs(i + j + 0.1 - c[i][j]) > epsilon)
		{
			printf("check faild. c[%d][%d]=%f\n", i, j, c[i][j]);
			return;
		}
	}

	printf("check success. c[%d][%d]=%f\n", N - 1, N - 1, c[N - 1][N - 1]);
	return;
}

int main()
{
//	print_parameters();

//	print_utilization();

	if (init_and_malloc() != 0) {
		printf("init_and_malloc() failed.\n");
		cuda_print(status);
		return 0;
	}

	test_serial(); // just to load the memory

	begin = GET_CURRENT_MICRO;
	test_parallel();
	end = GET_CURRENT_MICRO;
	printf("cudaExecute parallel cost = %llu\n", end - begin);

	begin = GET_CURRENT_MICRO;
	test_serial();
	end = GET_CURRENT_MICRO;
	printf("cudaExecute serial cost = %llu\n", end - begin);

	copy_and_check();
	deinit_and_free();
	return 0;
}
