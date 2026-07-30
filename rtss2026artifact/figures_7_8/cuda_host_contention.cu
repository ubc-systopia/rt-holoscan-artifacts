#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

struct Config {
  bool malloc_workload = true;
  bool matmul_workload = true;
  bool green_contexts = false;
  size_t malloc_size = 1024;
  int malloc_calls = 100;
  int matmul_size = 512;
  int matmul_calls = 100;
  int repetitions = 1000;
  int sm_partition = 0;
  std::string output;
};

class Barrier {
 public:
  explicit Barrier(int participants) : participants_(participants) {}
  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    const int generation = generation_;
    if (++arrived_ == participants_) {
      arrived_ = 0;
      ++generation_;
      condition_.notify_all();
      return;
    }
    condition_.wait(lock, [&] { return generation_ != generation; });
  }
 private:
  int participants_, arrived_ = 0, generation_ = 0;
  std::mutex mutex_;
  std::condition_variable condition_;
};

__global__ void sgemm_naive(int n, const float* a, const float* b, float* c) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n || col >= n) return;
  float sum = 0;
  for (int i = 0; i < n; ++i) sum += a[row * n + i] * b[i * n + col];
  c[row * n + col] = sum;
}

[[noreturn]] void fail(const char* where, const char* detail) {
  std::cerr << where << ": " << detail << '\n';
  std::exit(EXIT_FAILURE);
}
void check(cudaError_t result, const char* where) {
  if (result != cudaSuccess) fail(where, cudaGetErrorString(result));
}
void check_driver(CUresult result, const char* where) {
  if (result != CUDA_SUCCESS) {
    const char* name = "unknown";
    cuGetErrorName(result, &name);
    fail(where, name);
  }
}

struct ThreadResult { double average_ms = 0, maximum_ms = 0; };

struct GreenResources {
  CUdevice device{};
  std::vector<CUdevResource> partitions;
};

GreenResources make_partitions(int threads, int sm_partition) {
  GreenResources resources;
  check_driver(cuDeviceGet(&resources.device, 0), "cuDeviceGet");
  CUdevResource all_sms{};
  check_driver(cuDeviceGetDevResource(resources.device, &all_sms,
                                      CU_DEV_RESOURCE_TYPE_SM),
               "cuDeviceGetDevResource");
  unsigned int groups = threads;
  resources.partitions.resize(threads);
  check_driver(cuDevSmResourceSplitByCount(resources.partitions.data(), &groups,
                                            &all_sms, nullptr, 0, sm_partition),
               "cuDevSmResourceSplitByCount");
  if (groups != static_cast<unsigned int>(threads))
    fail("Green Context partitioning", "GPU lacks enough SMs for this run");
  return resources;
}

ThreadResult worker(const Config& config, int worker_id, Barrier& start,
                    const GreenResources* resources, int samples_per_repeat) {
  cudaStream_t stream{};
  CUgreenCtx green{};
  if (resources) {
    CUdevResourceDesc descriptor{};
    const auto& partition = resources->partitions.at(worker_id);
    check_driver(cuDevResourceGenerateDesc(&descriptor, &partition, 1),
                 "cuDevResourceGenerateDesc");
    check_driver(cuGreenCtxCreate(&green, descriptor, resources->device,
                                  CU_GREEN_CTX_DEFAULT_STREAM),
                 "cuGreenCtxCreate");
    CUcontext context{};
    check_driver(cuCtxFromGreenCtx(&context, green), "cuCtxFromGreenCtx");
    check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    check_driver(cuGreenCtxStreamCreate(&stream, green, CU_STREAM_NON_BLOCKING, 0),
                 "cuGreenCtxStreamCreate");
  } else {
    check(cudaStreamCreate(&stream), "cudaStreamCreate");
  }

  float *a = nullptr, *b = nullptr, *c = nullptr;
  const size_t bytes = static_cast<size_t>(config.matmul_size) * config.matmul_size * sizeof(float);
  dim3 block(16, 16);
  dim3 grid((config.matmul_size + 15) / 16, (config.matmul_size + 15) / 16);
  if (config.matmul_workload) {
    check(cudaMallocAsync(&a, bytes, stream), "cudaMallocAsync(a)");
    check(cudaMallocAsync(&b, bytes, stream), "cudaMallocAsync(b)");
    check(cudaMallocAsync(&c, bytes, stream), "cudaMallocAsync(c)");
    check(cudaMemsetAsync(a, 0, bytes, stream), "cudaMemsetAsync(a)");
    check(cudaMemsetAsync(b, 0, bytes, stream), "cudaMemsetAsync(b)");
    sgemm_naive<<<grid, block, 0, stream>>>(config.matmul_size, a, b, c);
    check(cudaGetLastError(), "kernel warmup launch");
    check(cudaStreamSynchronize(stream), "kernel warmup synchronization");
  }

  std::vector<double> timings;
  timings.reserve(samples_per_repeat);
  start.wait();
  for (int sample = 0; sample < samples_per_repeat; ++sample) {
    const auto begin = std::chrono::steady_clock::now();
    if (config.malloc_workload) {
      for (int i = 0; i < config.malloc_calls; ++i) {
        void* pointer = nullptr;
        check(cudaMallocAsync(&pointer, config.malloc_size, stream), "cudaMallocAsync");
        check(cudaFreeAsync(pointer, stream), "cudaFreeAsync");
      }
    }
    if (config.matmul_workload) {
      for (int i = 0; i < config.matmul_calls; ++i) {
        sgemm_naive<<<grid, block, 0, stream>>>(config.matmul_size, a, b, c);
      }
      check(cudaGetLastError(), "kernel launch");
    }
    const auto end = std::chrono::steady_clock::now();
    timings.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
  }

  if (config.matmul_workload) {
    check(cudaFreeAsync(a, stream), "cudaFreeAsync(a)");
    check(cudaFreeAsync(b, stream), "cudaFreeAsync(b)");
    check(cudaFreeAsync(c, stream), "cudaFreeAsync(c)");
  }
  check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  if (green) check_driver(cuGreenCtxDestroy(green), "cuGreenCtxDestroy");
  return {std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size(),
          *std::max_element(timings.begin(), timings.end())};
}

void run(const Config& config) {
  std::ofstream output(config.output);
  if (!output) fail("output", "cannot open CSV file");
  output << "Experiment,Threads,AvgExecTime,MaxExecTime\n";
  const std::vector<int> thread_counts = config.green_contexts
      ? std::vector<int>{1, 2, 3, 4} : std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  // The non-GC original benchmark takes the maximum of 50 timed sequences per
  // repetition. The GC result stores one sequence/repetition; the plot groups
  // ten repetitions before drawing each boxplot observation.
  const int samples = config.green_contexts ? 1 : 50;
  for (int repetition = 1; repetition <= config.repetitions; ++repetition) {
    for (int threads : thread_counts) {
      GreenResources green;
      const GreenResources* green_ptr = nullptr;
      if (config.green_contexts) {
        green = make_partitions(threads, config.sm_partition);
        green_ptr = &green;
      }
      Barrier start(threads);
      std::vector<std::thread> workers;
      std::vector<ThreadResult> results(threads);
      for (int worker_id = 0; worker_id < threads; ++worker_id) {
        workers.emplace_back([&, worker_id] {
          results[worker_id] = worker(config, worker_id, start, green_ptr, samples);
        });
      }
      for (auto& thread : workers) thread.join();
      double average = 0, maximum = 0;
      for (const auto& result : results) {
        average += result.average_ms;
        maximum = std::max(maximum, result.maximum_ms);
      }
      output << repetition << ',' << threads << ',' << average / threads << ',' << maximum << '\n';
    }
  }
}

void usage() {
  std::cout << "Usage: cuda_host_contention --output FILE [options]\n"
            << "  --malloc-only | --matmul-only\n"
            << "  --malloc-size BYTES --matmul-size N --calls N --repetitions N\n"
            << "  --green-contexts --sm-partition N\n";
}

int main(int argc, char** argv) {
  Config config;
  static option options[] = {{"output", required_argument, nullptr, 'o'},
      {"malloc-only", no_argument, nullptr, 'a'}, {"matmul-only", no_argument, nullptr, 'b'},
      {"malloc-size", required_argument, nullptr, 'm'}, {"matmul-size", required_argument, nullptr, 'n'},
      {"calls", required_argument, nullptr, 'c'}, {"repetitions", required_argument, nullptr, 'r'},
      {"green-contexts", no_argument, nullptr, 'g'}, {"sm-partition", required_argument, nullptr, 's'},
      {"help", no_argument, nullptr, 'h'}, {nullptr, 0, nullptr, 0}};
  for (int option; (option = getopt_long(argc, argv, "o:abm:n:c:r:gs:h", options, nullptr)) != -1;) {
    switch (option) {
      case 'o': config.output = optarg; break;
      case 'a': config.matmul_workload = false; break;
      case 'b': config.malloc_workload = false; break;
      case 'm': config.malloc_size = std::stoull(optarg); break;
      case 'n': config.matmul_size = std::stoi(optarg); break;
      case 'c': config.malloc_calls = config.matmul_calls = std::stoi(optarg); break;
      case 'r': config.repetitions = std::stoi(optarg); break;
      case 'g': config.green_contexts = true; break;
      case 's': config.sm_partition = std::stoi(optarg); break;
      default: usage(); return option == 'h' ? 0 : 1;
    }
  }
  if (config.output.empty() || (!config.malloc_workload && !config.matmul_workload) ||
      (config.green_contexts && config.sm_partition <= 0)) {
    usage(); return 1;
  }
  if (config.green_contexts) check_driver(cuInit(0), "cuInit");
  run(config);
}
