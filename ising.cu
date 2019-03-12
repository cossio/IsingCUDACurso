/* Monte Carlo simulation of the Ising model using CUDA */
/* Author: Jorge Fernandez de Cossio Diaz */
/* March, 2019 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cassert>

#include <cuda.h>
#include <curand_kernel.h>
//#include <curand.h>

#define RANDSEED 5  // random seed

/* Linear dimension of square grid. The block-size in my laptop is 1024. 
Therefore we set L=32 so that there are a total of L^2 = 1024 spins. */
#define L 32	// linear dimension of square grid.
#define N (L*L)	// total number of spins

#define ITERATIONS 10001	// number of iterations

/* linear index of current block */
__device__ int globalBlockIdx() {
    return blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
}

/* linear index of current thread inside its block */
__device__ int threadIdxInBlock() {
	return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
}

/* total number of threads per block */
__device__ int blockVolume() {
    return blockDim.x * blockDim.y * blockDim.z;
}

/* gobal linear index of current thread */
__device__ int globalThreadIdx() {
    return threadIdxInBlock() + globalBlockIdx() * blockVolume();
}

/* setups random number generation in each thread */
__global__ void initialize_rand(curandState *rngState) {
	// Each thread gets same seed, a different sequence number, no offset
	int idx = globalThreadIdx();
	assert(0 <= idx && idx < N);
	curand_init(RANDSEED, idx, 0, rngState + idx);
}

/* random uniform real in [0,1] */
__device__ float randreal(curandState *rngState) {
	return curand_uniform(rngState + globalThreadIdx());
}

/* returns random -1 or +1 */
__device__ short randspin(curandState *rngState) {
	return 2 * (short)roundf(randreal(rngState)) - 1;
}

/* returns linear index corresponding to Cartesian index x, y 
in the periodic square grid */
__host__ __device__ short linear_index(int x, int y) {
	while (x < 0)  {x += L; }
	while (x >= L) {x -= L; }
	while (y < 0)  {y += L; }
	while (y >= L) {y -= L; }
	assert(x >= 0 && x < L);
	assert(y >= 0 && y < L);
	int idx = x + L * y;
	assert(0 <= idx && idx < N);
	return idx;
}

__host__ __device__ short get(short *spins, int x, int y) {
	return spins[linear_index(x, y)];
}

__host__ __device__ short set(short *spins, int x, int y, short state) {
	spins[linear_index(x, y)] = state;
	return state;
}

/* initializes the spins to random states */
__global__ void initialize_spins(short *spins, curandState *rngState) {
	spins[globalThreadIdx()] = randspin(rngState);
}

/* sum of neighboring spins */
__device__ short neighbor_sum(short *spins) {
	int x = threadIdx.x;
	int y = threadIdx.y;
	short sum = 0;
	sum += get(spins, x - 1, y);
	sum += get(spins, x + 1, y);
	sum += get(spins, x, y - 1);
	sum += get(spins, x, y + 1);
	return sum;
}

/* update all spins with metropolis rule at inverse temperature beta. */
__device__ void metropolis(short *spins, curandState *rngState, float beta) {
	int idx = globalThreadIdx();
	short state = spins[idx];
	assert(state == 1 || state == -1);
	
	// Metropolis update rule
	float deltaE = 2.0f * state * neighbor_sum(spins);
	//if (idx == 10) { printf("thread %i, beta %f, deltaE %f, expf(-beta * deltaE) %f\n", idx, beta, deltaE, expf(-beta * deltaE)); }
	//if (deltaE <= 0) { assert(randreal(rngState) < expf(-beta * deltaE)); }
	float u = randreal(rngState);
	if (u < 0.5 && u < expf(-beta * deltaE)) {
		state = -state;
	}

	// synchronous update
	__syncthreads();	// wait for all threads to compute new state
	spins[idx] = state;
	__syncthreads();	// wait for all threads to update spins
}

/* copies spin states between two arrays in device memory */
__device__ void spinsCpy(short *to, short *from) {
	int idx = globalThreadIdx();
	to[idx] = from[idx];
	__syncthreads();
}

/* simulates the system of spins in shared memory */
__global__ void simulate(short *spinsGlob, curandState *rngState, float beta) {
	__shared__ short spinsShared[N];
	// copy spins from global to shared memory
	spinsCpy(spinsShared, spinsGlob);
	
	// simulate
	for (int iter = 0; iter < ITERATIONS; ++iter) {
		metropolis(spinsShared, rngState, beta);
	}

	// copy spins back to global memory
	spinsCpy(spinsGlob, spinsShared);
}

/* return magnetization of system of spins */
__host__ __device__ float magnetization(short* spins) {
	short M = 0;
	for (int i = 0; i < N; ++i) {
		assert(spins[i] == -1 || spins[i] == 1);
		M += spins[i];
	}
	return (float)M / N;
}

/* prints the grid of spins */
__host__ __device__ void print_spins(short *spins) {
	for (int x = 0; x < L; ++x) {
		for (int y = 0; y < L; ++y) {
			short s = get(spins, x, y);
			if (s == 1) {
				printf("+ ");
			} else if (s == -1) {
				printf("- ");
			} else {
				printf("%i ", s);
			}
		}
		printf("\n");
	}
}

int main(void) {
	printf("Simulating %i spins, on a square grid of length %i\n", N, L);

	// random setup
	curandState *rngStatesDev;
	cudaMalloc(&rngStatesDev, N * sizeof(curandState));

	dim3 blockSize(L,L);
	initialize_rand<<<1, blockSize>>>(rngStatesDev);

	// allocate host/device memory for spins
	short *spins;
	cudaMallocManaged(&spins, N * sizeof(short));

	// initialize spins to random configurations
	initialize_spins<<<1, blockSize>>>(spins, rngStatesDev);
	cudaDeviceSynchronize();

	//printf("Initialized spins (in random state), |m| = %f\n", abs(magnetization(spins)));
	//print_spins(spins);
	
	// simulate
	printf("beta\tabsolute magnetization\n");
	for (float beta = 0.0f; beta <= 1.0f; beta += 0.01f) {
		simulate<<<1, blockSize>>>(spins, rngStatesDev, beta);
		cudaDeviceSynchronize();

		float m = magnetization(spins);
		printf("%f\t%f\n", beta, abs(m));
		fflush(stdout);
	}

	//printf("Final configuration, |m| = %f\n", abs(magnetization(spins)));
	//print_spins(spins);

	cudaFree(rngStatesDev);
	cudaFree(spins);

	return 0;
}