#include <cuda_runtime_api.h>
#include "workflow/SubTask.h"

#define cuda_print(X) \
{\
	cudaError_t err = X;\
	fprintf(stderr, "[cuda_print] %s status=%d, %s\n",\
	#X, err, cudaGetErrorName(err));\
}

class CudaRequest : public SubTask
{
public:
	CudaRequest(cudaStream_t stream)
	{
		this->stream = stream;
	}

public:
	virtual void dispatch()
	{
		cudaError_t status;

		status = cudaStreamAddCallback(this->stream, CudaRequest::callback, this, 0);

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
		CudaRequest *request = (CudaRequest *)data;

		request->status = status;
		request->subtask_done();
	}
};

