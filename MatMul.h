#ifndef __CUDA_MATMUL_H__
#define __CUDA_MATMUL_H__

#include "CudaTask.h"
//#include "workflow/WFGlobal.h"

#define cuda_print(X) \
{\
	cudaError_t err = X;\
	fprintf(stderr, "[cuda_print] %s status=%d, %s\n",\
	#X, err, cudaGetErrorName(err));\
}

__global__ void matrix_mul_gpu(int *a, int* b, int* c, int width)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
                
    int sum = 0;
    for (int k = 0; k < width; k++)
    {
        int tmp_a = a[j * width + k];
        int tmp_b = b[k * width + i];
        sum += tmp_a * tmp_b;
    }
    c[j * width + i] = sum;
}

/*
template<class INPUT, class OUTPUT>
class CudaMemcpyTask : public WFCudaTask<INPUT, OUTPUT>
{
public:
	CudaMemcpyTask(cudaStream_t stream, )
};
*/

template<class INPUT, class OUTPUT>
class CudaMatMulTask : public WFCudaTask<INPUT, OUTPUT>
{
public:
	CudaMatMulTask(dim3 grid, dim3 block,
				   std::function<void (WFThreadTask<INPUT, OUTPUT> *)>&& cb) :
		WFCudaTask<INPUT, OUTPUT>(this->stream, std::move(cb))
	{
		this->grid = grid;
		this->block = block;
		this->stream = nullptr;
	}

	virtual ~CudaMatMulTask()
	{
		cudaStreamDestroy(this->stream);
	}

	virtual void dispatch();

	virtual SubTask *done();

	INPUT *get_device_input() { return &this->device_input; }
	OUTPUT *get_device_output() { return &this->device_output; }

private:
	cudaStream_t stream;
	INPUT device_input;
	OUTPUT device_output;
	dim3 grid;
	dim3 block;
};

template<class INPUT, class OUTPUT>
void CudaMatMulTask<INPUT, OUTPUT>::dispatch()
{
	INPUT *in = &this->input;
	OUTPUT *out = &this->output;

	INPUT *d_in = &this->device_input;
	OUTPUT *d_out = &this->device_output;
	d_in->init(in->row, in->col);
	d_out->init(out->row, out->col);

	int size = in->row * in->col * sizeof(int);

	cudaMalloc((void**)&d_in->a, size);	
	cudaMalloc((void**)&d_in->b, size);
	cudaMalloc((void**)&d_out->c, size);

//	cudaMemcpyAsync(d_in->a, in->a, size, cudaMemcpyHostToDevice, this->stream);
	cudaMemcpy(d_in->a, in->a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in->b, in->b, size, cudaMemcpyHostToDevice);

//	cuda_print(cudaStreamCreate(&this->stream));
//	cuda_print(cudaStreamQuery(this->stream));

	matrix_mul_gpu <<< this->grid, this->block, 0, this->stream >>>
				(d_in->a, d_in->b, d_out->c, in->col);

	this->WFCudaTask<INPUT, OUTPUT>::dispatch();
	return;
/*
	// other error
	this->status = cudaErrorMemoryAllocation;
	this->subtask_done();
	return;
*/
}

template<class INPUT, class OUTPUT>
SubTask *CudaMatMulTask<INPUT, OUTPUT>::done()
{
	SeriesWork *series = series_of(this);

	OUTPUT *out = &this->output;
	INPUT *d_in = &this->device_input;
	OUTPUT *d_out = &this->device_output;

	int size = out->row * out->col * sizeof(int);

	cudaMemcpy(out->c, d_out->c, size, cudaMemcpyDeviceToHost);

	cuda_print(cudaStreamQuery(this->stream));

	cudaFree(d_in->a);
	cudaFree(d_in->b);
	cudaFree(d_out->c);

	if (this->callback)
		this->callback(this);

	delete this;
	return series->pop();
}

template<class INPUT, class OUTPUT>
class CudaMatMulAsyncTask : public CudaAsyncTask<INPUT, OUTPUT>
{
public:
	CudaMatMulAsyncTask(dim3 grid, dim3 block,
			   		   std::function<void (CudaAsyncTask<INPUT, OUTPUT> *)>&& cb) :
			CudaAsyncTask<INPUT, OUTPUT>(this->stream, std::move(cb))
	{
		this->grid = grid;
		this->block = block;
		this->stream = nullptr;
		this->start_event = nullptr;
		this->end_event = nullptr;
	}

	virtual ~CudaMatMulAsyncTask()
	{
		cudaStreamDestroy(this->stream);
		cudaEventDestroy(this->start_event);
		cudaEventDestroy(this->end_event);
	}

	virtual void dispatch();

	virtual SubTask *done();

	INPUT *get_device_input() { return &this->device_input; }
	OUTPUT *get_device_output() { return &this->device_output; }

private:
	cudaStream_t stream;
	INPUT device_input;
	OUTPUT device_output;
	dim3 grid;
	dim3 block;
	cudaEvent_t start_event, end_event; // for debug
};

template<class INPUT, class OUTPUT>
void CudaMatMulAsyncTask<INPUT, OUTPUT>::dispatch()
{
	INPUT *in = &this->input;
	OUTPUT *out = &this->output;

	INPUT *d_in = &this->device_input;
	OUTPUT *d_out = &this->device_output;
	d_in->init(in->row, in->col);
	d_out->init(out->row, out->col);

	int size = in->row * in->col * sizeof(int);

	cudaMalloc((void**)&d_in->a, size);	
	cudaMalloc((void**)&d_in->b, size);
	cudaMalloc((void**)&d_out->c, size);

	cudaMemcpy(d_in->a, in->a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in->b, in->b, size, cudaMemcpyHostToDevice);

	cuda_print(cudaStreamCreate(&this->stream));

	matrix_mul_gpu <<< this->grid, this->block, 0, this->stream >>>
				(d_in->a, d_in->b, d_out->c, in->col);

	fprintf(stderr, "dispatch() stream_addr %lu\n", this->stream);
	this->CudaAsyncTask<INPUT, OUTPUT>::dispatch();
	return;
/*
	// other error
	this->status = cudaErrorMemoryAllocation;
	this->subtask_done();
	return;
*/
}

template<class INPUT, class OUTPUT>
SubTask *CudaMatMulAsyncTask<INPUT, OUTPUT>::done()
{
	SeriesWork *series = series_of(this);

	OUTPUT *out = &this->output;

	INPUT *d_in = &this->device_input;
	OUTPUT *d_out = &this->device_output;

	int size = out->row * out->col * sizeof(int);

	cuda_print(cudaMemcpy(out->c, d_out->c, size, cudaMemcpyDeviceToHost));

	cuda_print(cudaEventSynchronize(this->end_event));
	float runtime = 0.0f;
	cudaEventElapsedTime(&runtime, this->start_event, this->end_event);
	fprintf(stderr, "get runtime from event: %ls micro seconds\n", runtime);

	cudaFree(d_in->a);
	cudaFree(d_in->b);
	cudaFree(d_out->c);

	return this->CudaAsyncTask<INPUT, OUTPUT>::done();
}

// Factory start

template<class INPUT, class OUTPUT>
using cuda_callback = std::function<void (WFThreadTask<INPUT, OUTPUT> *)>;

template<class INPUT, class OUTPUT>
using cuda_async_callback = std::function<void (CudaAsyncTask<INPUT, OUTPUT> *)>;

class CudaTaskFactory
{
public:
	template<class INPUT, class OUTPUT, class CB = cuda_callback<INPUT, OUTPUT>>
	static CudaMatMulTask<INPUT, OUTPUT> *create_matmul_task(dim3 grid,
															 dim3 block,
															 CB callback)
	{
		auto *task = new CudaMatMulTask<INPUT, OUTPUT>(grid, block,
													   std::move(callback));
		return task;
	}

	template<class INPUT, class OUTPUT, class CB = cuda_async_callback<INPUT, OUTPUT>>	
	static  CudaMatMulAsyncTask<INPUT, OUTPUT> *create_matmul_async_task(dim3 grid,
																		 dim3 block,
																		 CB callback)
	{
		auto *task = new CudaMatMulAsyncTask<INPUT, OUTPUT>(grid, block,
															std::move(callback));
		return task;
	}
};


#endif
