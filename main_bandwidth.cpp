#ifdef USE_64BIT 
#define Float double
#define Int long long
#elif USE_32BIT
#define Float float
#define Int long long
#endif

#define NGPU 4
#define NTIMES 40
#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <time.h>

__global__ 
void copy_vector_kernel(Float* dest, Float* src, const Int m)
{
    for(Int i = threadIdx.x+64*blockIdx.x; i < m; i += 64*gridDim.x)
        dest[i] = src[i];
}

void openmp_copy_cuda(Float* vec, const Int& m)
{
    Float* src[NGPU];
    Float* dest[NGPU];
    omp_set_num_threads(NGPU);
    #pragma omp parallel
    {
        int id = omp_get_thread_num() % NGPU;
        cudaSetDevice(id);
        cudaMalloc(&src[id], sizeof(Float)*m);
        cudaMalloc(&dest[id], sizeof(Float)*m);
        cudaMemcpy(src[id], vec, sizeof(Float)*m, cudaMemcpyHostToDevice);
    }
    //double times;
    //clock_t start = clock(), diff;
    double start = omp_get_wtime();
    for(int i = 0; i < NTIMES; ++i)
    {
        #pragma omp parallel
        {
            int id = omp_get_thread_num() % NGPU;
            cudaSetDevice(id);
            copy_vector_kernel<<<4096, 64>>>(dest[(id+1)%NGPU], src[id], m);
            cudaDeviceSynchronize();
        }
    }
    double end = omp_get_wtime();
    //diff = clock() - start;
    double cpu_time_used = end-start;
    //times = cpu_time_used;
    std::cout << "OpenMP Profile:\n";
    std::cout << "Ave. Time: " << cpu_time_used/NTIMES << "s "<< std::endl;
    std::cout << "Band Width: " << NTIMES*sizeof(Float)*m*2/cpu_time_used/(1024*1024*1024) << " GB/s" << std::endl;
/*
    double max_time = 0;
    double ave_time = 0;
    double min_time = 0;
    for(int i = 0; i < NTIMES; ++i)
        if(max_time < times[i])
            max_time = times[i];

    for(int i = 0; i < NTIMES; ++i)
        ave_time += times[i]/NGPU;

    min_time = times[0];

    for(int i = 1; i < NTIMES; ++i)
        if(min_time > times[i])
            min_time = times[i];

    std::cout << "OpenMP Profile:\n";
    std::cout << max_time << "s, " << ave_time << "s, " << min_time << "s.\n";
*/
    for(int i = 0; i < NGPU; ++i)
    {
        cudaSetDevice(i);
        cudaFree(src[i]);
        cudaFree(dest[i]);
    }
}

void single_copy_cuda(Float* vec, const Int& m)
{
    Float* src[NGPU];
    Float* dest[NGPU];
    omp_set_num_threads(NGPU);
    #pragma omp parallel
    {
        int id = omp_get_thread_num() % NGPU;
        cudaSetDevice(id);
        cudaMalloc(&src[id], sizeof(Float)*m);
        cudaMalloc(&dest[id], sizeof(Float)*m);
        cudaMemcpy(src[id], vec, sizeof(Float)*m, cudaMemcpyHostToDevice);
    }
    //double times;
    //clock_t start = clock(), diff;
    double start = omp_get_wtime();

    for(int i = 0; i < NTIMES; ++i)
    {
        //clock_t start = clock(), diff;
        for(int g = 0; g < NGPU; ++g)
        {
            cudaSetDevice(g);
            copy_vector_kernel<<<4096, 64>>>(dest[(g+1)%NGPU], src[g], m);        
        }
        for(int g = 0; g < NGPU; ++g)
        {
            cudaSetDevice(g);
            cudaDeviceSynchronize();
        } 
        //diff = clock() - start;
        //double cpu_time_used = ((double)diff) / CLOCKS_PER_SEC;
        //times[i] = cpu_time_used;
    }
    double end = omp_get_wtime();
    double cpu_time_used = end-start;

    //diff = clock() - start;
    //double cpu_time_used = ((double)diff) / CLOCKS_PER_SEC;
    //times = cpu_time_used;
/*
    double max_time = 0;
    double ave_time = 0;
    double min_time = 0;
    for(int i = 0; i < NTIMES; ++i)
        if(max_time < times[i])
            max_time = times[i];

    for(int i = 0; i < NTIMES; ++i)
        ave_time += times[i]/NGPU;

    min_time = times[0];

    for(int i = 1; i < NTIMES; ++i)
        if(min_time > times[i])
            min_time = times[i];
    */
    std::cout << "Single Profile:\n";
    std::cout << "Ave. Time: " << cpu_time_used/NTIMES << "s "<< std::endl;
    std::cout << "Band Width: " << NTIMES*sizeof(Float)*m*2/cpu_time_used/(1024*1024*1024) << " GB/s" << std::endl;
    //std::cout << max_time << "s, " << ave_time << "s, " << min_time << "s.\n";
    for(int i = 0; i < NGPU; ++i)
    {
        cudaSetDevice(i);
        cudaFree(src[i]);
        cudaFree(dest[i]);
    }
}

void local_copy_cuda(Float* vec, const Int& m)
{
    cudaSetDevice(0);
    Float* src, *dest;
    //double times[NTIMES];

    cudaMalloc(&src, sizeof(Float)*m);
    cudaMalloc(&dest, sizeof(Float)*m);

    cudaMemcpy(src, vec, sizeof(Float)*m, cudaMemcpyHostToDevice);

    //clock_t start = clock(), diff;
    double start = omp_get_wtime();
    for(int i = 0; i < NTIMES; ++i)
    {
        copy_vector_kernel<<<4096, 64>>>(dest, src, m);
        cudaDeviceSynchronize();
        //diff = clock() - start;
        //double cpu_time_used = ((double)diff) / CLOCKS_PER_SEC;
        //times[i] = cpu_time_used;
    } 
    double end = omp_get_wtime();
    double cpu_time_used = end-start;

    //diff = clock() - start;
    //double cpu_time_used = ((double)diff) / CLOCKS_PER_SEC;
/*
    double max_time = 0;
    double ave_time = 0;
    double min_time = 0;
    for(int i = 0; i < NTIMES; ++i)
        if(max_time < times[i])
            max_time = times[i];

    for(int i = 0; i < NTIMES; ++i)
        ave_time += times[i]/NGPU;

    min_time = times[0];

    for(int i = 1; i < NTIMES; ++i)
        if(min_time > times[i])
            min_time = times[i];
*/

    std::cout << "Single Local Copy Profile:\n";
    std::cout << "Ave. Time: " << cpu_time_used/NTIMES << "s "<< std::endl;
    std::cout << "Band Width: " << NTIMES*sizeof(Float)*m*2/cpu_time_used/(1024*1024*1024) << " GB/s" << std::endl;
    //std::cout << max_time << "s, " << ave_time << "s, " << min_time << "s.\n";
    cudaFree(src);
    cudaFree(dest);
}

int main(int argc, char** argv)
{
    Int m = (Int)atoll(argv[1]);

    Float* vec = new Float[m];
    for(Int i = 0; i < m; ++i)
        vec[i] = (Float)rand()/RAND_MAX;

    for(int i = 0; i < NGPU; ++i)
    {
        cudaSetDevice(i);
        for(int j = 0; j < NGPU; ++j)
            if(i != j)
                cudaDeviceEnablePeerAccess(j, 0);
    }

    openmp_copy_cuda(vec, m);
    single_copy_cuda(vec, m);
    local_copy_cuda(vec, m);

    return 0;
}


