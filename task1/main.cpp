#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include <CL/cl.hpp>

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>

int main() {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {
        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("/tmp/tmp.ULjB9xJYTi/reduce.c");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        try {
            program.build(devices);
        }
        catch (cl::Error const &e) {
            std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
            std::cout << log_str;
            return 0;
        }

        size_t const block_size = 16;


        std::ifstream input("/tmp/tmp.ULjB9xJYTi/input.txt");
        int n, m;
        input >> n >> m;


        std::vector<float> a(n * n);
        std::vector<float> b(m * m);
        std::vector<float> c(n * n, 1);


        for (int i = 0; i < n * n; ++i)
            input >> a[i];

        for (int i = 0; i < m * m; ++i)
            input >> b[i];


        // allocate device buffer to hold message
        cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(float) * n * n);
        cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(float) * m * m);
        cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * n * n);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * n * n, &a[0]);
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(float) * m * m, &b[0]);

        // load named kernel from opencl source
        cl::Kernel kernel_gmem(program, "conv");
        kernel_gmem.setArg(0, dev_a);
        kernel_gmem.setArg(1, dev_b);
        kernel_gmem.setArg(2, dev_c);
        kernel_gmem.setArg(3, n);
        kernel_gmem.setArg(4, m);
        kernel_gmem.setArg(5, (m - 1) / 2);

        size_t thread_count = (n / block_size + 1) * block_size;
        queue.enqueueNDRangeKernel(kernel_gmem, cl::NullRange, cl::NDRange(thread_count, thread_count),
                                   cl::NDRange(block_size, block_size));


        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(float) * n * n, &c[0]);

        for (int i = 0; i < n * n; ++i) {
            std::cout << c[i] << " ";
            if ((i + 1) % n == 0)
                std::cout << std::endl;
        }

    }
    catch (cl::Error const &e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}