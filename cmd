nvcc -o cuda_demo matrix_mul_demo.cu -g \
		-L/usr/local/cuda-9.0/targets/x86_64-linux/lib/ \
		-I/usr/local/cuda-9.0/targets/x86_64-linux/include/ \
		-L/root/subtask/_lib/ \
		-I/root/subtask/_include/ \
		-lcudart -lworkflow \
		--std=c++11

export LD_LIBRARY_PATH=/root/subtask/_lib/:$LD_LIBRARY_PATH
