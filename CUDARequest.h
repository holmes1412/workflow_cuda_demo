#include <cuda_runtime_api.h>
#include "workflow/SubTask.h"

#define cuda_print(X) \
{\
	cudaError_t err = X;\
	fprintf(stderr, "[cuda_print] %s status=%d, %s\n",\
	#X, err, cudaGetErrorName(err));\
}

class CUDARequest : public SubTask
{
public:
	CUDARequest(cudaStream_t stream)
	{
		this->stream = stream;
	}

public:
	virtual void dispatch()
	{
		cudaError_t status;

		fprintf(stderr, "stream addr %lu\n", this->stream);
		status = cudaStreamAddCallback(this->stream, CUDARequest::callback, this, 0);
//		this->subtask_done();

		if (status != cudaSuccess)
		{
			this->status = status;
			this->subtask_done();
		}
	}

protected:
	cudaError_t status;

protected:
	cudaStream_t stream;

protected:
	static void callback(cudaStream_t stream, cudaError_t status, void *data)
	{
		CUDARequest *request = (CUDARequest *)data;
		fprintf(stderr, "stream addr %lu\n", request->stream);

		request->status = status;
		request->subtask_done();
	}
};

