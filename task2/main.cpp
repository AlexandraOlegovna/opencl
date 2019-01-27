// https://tmramalho.github.io/blog/2014/06/23/parallel-programming-with-opencl-and-python-parallel-scan/

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <assert.h>
#include <string>


size_t const BLOCK = 256;


std::vector<float> iterate(std::vector<float> const &input);

int main(int argc, char** argv) {

    std::ifstream in("/tmp/tmp.uXIp4wdVn0/input.txt");
    int n;
    in >> n;

    std::vector<float> input(n);

    for (int i = 0; i < n; ++i)
        in >> input[i];

    input.resize(input.size() + (BLOCK - input.size() % BLOCK));


    std::vector<float> sum = iterate(input);

    for (int i = 0; i < n; ++i) {
        std::cout << std::fixed;
        std::cout << std::setprecision(3);
        std::cout << sum[i] << " ";
    }

    return 0;
}

std::vector<float> iterate(std::vector<float> const &input) {
    std::vector <cl::Platform> platforms;
    std::vector <cl::Device> devices;
    std::vector <cl::Kernel> kernels;

    try {

        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("/tmp/tmp.uXIp4wdVn0/main.c");
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
            exit(0);
        }

        std::vector<float> output(input.size());


        cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(float) * input.size());
        cl::Buffer dev_b(context, CL_MEM_WRITE_ONLY, sizeof(float) * output.size());

        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * input.size(), &input[0]);

        queue.finish();

        cl::Kernel kernel_func(program, "scan");
        cl::KernelFunctor func_s(kernel_func, queue, cl::NullRange, cl::NDRange(input.size()), cl::NDRange(BLOCK));

        cl::Event ev = func_s(dev_a, dev_b, cl::__local(sizeof(float) * BLOCK), cl::__local(sizeof(float) * BLOCK));

        ev.wait();
        queue.enqueueReadBuffer(dev_b, CL_TRUE, 0, sizeof(float) * output.size(), &output[0]);

        if (input.size() < BLOCK + 1)
            return output;

        std::vector<float> sum;
        for (int i = 0; i < input.size() / BLOCK; ++i)
            sum.push_back(output[i * BLOCK + BLOCK - 1]);

        auto res = iterate(sum);

        cl::Buffer dev_c(context, CL_MEM_READ_ONLY, sizeof(float) * res.size());

        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * input.size(), &output[0]);
        queue.enqueueWriteBuffer(dev_c, CL_TRUE, 0, sizeof(float) * res.size(), &res[0]);
        queue.finish();

        cl::Kernel kernel_merge(program, "combine");
        cl::KernelFunctor func_m( kernel_merge, queue, cl::NullRange, cl::NDRange(input.size()), cl::NDRange(BLOCK));

        func_m(dev_a, dev_c, dev_b).wait();
        queue.enqueueReadBuffer(dev_b, CL_TRUE, 0, sizeof(float) * input.size(), &output[0]);

        return output;
    } catch (cl::Error e) {
        throw std::runtime_error(std::string(e.what()) + " : " + std::to_string(e.err()));
    }
}