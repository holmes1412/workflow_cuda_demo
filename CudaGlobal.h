#ifndef __CUDA_GLOBAL_H_
#define __CUDA_GLOBAL_H_

#include "workflow/Executor.h"
#include "workflow/WFGlobal.h"

//TODO: DEFAULT_CUDA_WAIT_THREADS

class CudaGlobal
{
public:
	static CudaGlobal *get_instance()
    {
        static CudaGlobal cuda_global_instance;
        return &cuda_global_instance;
    }

	static Executor *get_executor()
	{
		static Executor executor;
		
		return &get_instance()->executor;
	}

	static ExecQueue *get_queue()
	{
		return &get_instance()->queue;
	}

	CudaGlobal()
	{
		int cuda_wait_threads = 4;
/*
	    int cuda_wait_threads = g_cuda_setting.cuda_wait_threads;
        if (cuda_wait_threads <= 0)
            cuda_wait_threads = DEFAULT_CUDA_WAIT_THREADS;
*/
        int ret = this->executor.init(cuda_wait_threads);
        if (ret < 0)
            abort();
		this->queue.init();
	}

	~CudaGlobal()
	{
		this->executor.deinit();
	}

private:	
	Executor executor;
	ExecQueue queue;
};

#endif
