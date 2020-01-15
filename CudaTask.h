#ifndef __CUDA_TASK_H__
#define __CUDA_TASK_H__

#include <functional>
#include "CudaRequest.h"
#include "CudaGlobal.h"
#include "workflow/WFTask.h"
#include "workflow/Workflow.h"

template<class INPUT, class OUTPUT>
class CudaAsyncTask : public CudaRequest
{
public:
	void start()
	{
		assert(!series_of(this));
		Workflow::start_series_work(this, nullptr);
	}

	void dismiss()
	{
		assert(!series_of(this));
		delete this;
	}

public:
	INPUT *get_input() { return &this->input; }
	OUTPUT *get_output() { return &this->output; }

public:
	void *user_data;

public:
	int get_state() const { return this->state; }
//	int get_error() const { return this->error; }

public:
	void set_callback(std::function<void (CudaAsyncTask<INPUT, OUTPUT> *)> cb)
	{
		this->callback = std::move(cb);
	}

protected:
	virtual SubTask *done()
	{
		SeriesWork *series = series_of(this);

		cuda_print(cudaStreamQuery(this->stream));
		if (this->callback)
			this->callback(this);

		delete this;
		return series->pop();
	}

protected:
	INPUT input;
	OUTPUT output;
	std::function<void (CudaAsyncTask<INPUT, OUTPUT> *)> callback;

public:
	CudaAsyncTask(cudaStream_t stream,
			 	  std::function<void (CudaAsyncTask<INPUT, OUTPUT> *)>&& cb) :
		CudaRequest(stream),
		callback(std::move(cb))
	{
		this->user_data = NULL;
//		this->state = STATE_UNDEFINED;
//		this->error = 0;
	}

protected:
	virtual ~CudaAsyncTask() { }
};

template<class INPUT, class OUTPUT>
class WFCudaTask : public WFThreadTask<INPUT, OUTPUT>
{
public:
	virtual void execute()
	{
		cudaError_t status;
		status = cudaStreamSynchronize(this->stream);

		if (status == cudaSuccess)
			status = cudaStreamQuery(this->stream);
		cuda_print(status);

		this->status = status;
	}

	WFCudaTask(cudaStream_t stream,
			   std::function<void (WFThreadTask<INPUT, OUTPUT> *)>&& cb) :
			WFThreadTask<INPUT, OUTPUT>(CudaGlobal::get_instance()->get_queue(),
										CudaGlobal::get_instance()->get_executor(),
										std::move(cb))
	{
		this->stream = stream;
		this->user_data = NULL;
	}

protected:
	virtual ~WFCudaTask() { }

protected:
	cudaError_t status;
	cudaStream_t stream;
};

#endif
