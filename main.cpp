#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <stdint.h>
#include <random>
#include <chrono>
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

constexpr int N = 65535;

void print(uint64_t *vals, int max = N) {
    for (int i = 0; i < max; i++) {
        uint64_t cells = vals[i];
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                std::cout << ((cells & 1) ? '#' : '.');
                cells = cells >> 1;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

class LifeGrid {
    std::vector<uint64_t> regions;
    std::unordered_map<std::pair<int16_t, int16_t>, uint16_t> coords;
};

int main() {
    std::vector<cl::Platform> platforms;
    if (cl::Platform::get(&platforms) != CL_SUCCESS) {
        std::cerr << "Querying OpenCL platforms failed." << std::endl;
        return 1;
    }

    if (platforms.size() == 0) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        return 1;
    }

    auto platform = platforms.front();
    std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

    std::vector<cl::Device> devices;
    if (platform.getDevices(CL_DEVICE_TYPE_GPU, &devices) != CL_SUCCESS) {
        std::cerr << "Querying platform devices failed." << std::endl;
    }
    if (devices.size() == 0) {
        // No GPUs
        std::cerr << "No GPU found, switching to CPU." << std::endl; 
        devices.clear();
        if (platform.getDevices(CL_DEVICE_TYPE_ALL, &devices) != CL_SUCCESS) {
            std::cerr << "Querying platform devices failed." << std::endl;
            return 1;
        }

        if (devices.size() == 0) {
            std::cerr << "No devices found." << std::endl;
            return 1;
        }
    }

    for (auto device : devices) {
        std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    }

    auto device = devices.front();
    cl::Context context({device});
    cl::Program::Sources sources;

    std::ifstream cl_file("main.cl");
    if (!cl_file.is_open()) {
        std::cerr << "Failed to find main.cl." << std::endl;
        return 1;
    }
    std::ostringstream oss;
    oss << cl_file.rdbuf();
    std::string cl_contents = oss.str();

    sources.push_back(cl_contents);

    cl::Program program(context, sources);
    if (program.build() != CL_SUCCESS) {
        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return 1;
    }
    std::cout << "Warnings: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;

    std::cerr << "Built!" << std::endl;

    cl::Buffer buffer_regions_a(context, CL_MEM_READ_WRITE, sizeof(uint64_t) * N);
    cl::Buffer buffer_regions_b(context, CL_MEM_READ_WRITE, sizeof(uint64_t) * N);
    cl::Buffer buffer_neighbors(context, CL_MEM_READ_ONLY, sizeof(uint32_t) * N * 8);

    uint64_t cpu_regions_a[N];
    uint32_t cpu_neighbors[N * 8];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> rand_neighbor(0, N);
    std::uniform_int_distribution<uint64_t> rand_data(0, UINT64_MAX);

    // Initialize everything randomly.
    for (int i = 0; i < N; i++) {
        cpu_regions_a[i] = rand_data(gen);
        for (int j = 0; j < 8; j++) {
            cpu_neighbors[i*8 + j] = i + j;
        }
    }

    cl::CommandQueue queue(context, device);

    // Set up a zero region and a region with custom data.
    cpu_regions_a[0] = 0;
    cpu_regions_a[1] = (0b00000000'00000010'00000100'00000111'00000000'00000000'00000000'00000000);
    for (int i = 0; i < 16; i++) {
        cpu_neighbors[i] = 0;
    }

    print(cpu_regions_a, 2);

    queue.enqueueWriteBuffer(buffer_regions_a, CL_TRUE, 0, sizeof(uint64_t) * N, cpu_regions_a);
    queue.enqueueWriteBuffer(buffer_neighbors, CL_TRUE, 0, sizeof(uint32_t) * N * 8, cpu_neighbors);

    cl::Kernel sparse_life4(program, "sparse_life4");

    int GENERATIONS = 4;

    sparse_life4.setArg<int>(0, GENERATIONS);
    sparse_life4.setArg(2, buffer_neighbors);

    auto start = std::chrono::steady_clock::now();

    constexpr int n_iterations = 2000;
    for (int i = 0; i < n_iterations; i++) {
        sparse_life4.setArg(1, buffer_regions_a);
        sparse_life4.setArg(3, buffer_regions_b);
        queue.enqueueNDRangeKernel(sparse_life4, cl::NullRange, cl::NDRange(N), cl::NullRange);

        auto tmp = buffer_regions_a;
        buffer_regions_a = buffer_regions_b;
        buffer_regions_b = tmp;
    }

    queue.enqueueReadBuffer(buffer_regions_a, CL_TRUE, 0, sizeof(uint64_t) * N, cpu_regions_a);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    auto ms = elapsed.count();
    uint64_t cell_updates = N * 64ull * GENERATIONS * n_iterations;
    std::cout << "Execution time: " << elapsed.count() << "ms" << std::endl;
    std::cout << "CUps: " << cell_updates / (ms / 1000) << std::endl;

    print(cpu_regions_a, 10);

    return 0;
}