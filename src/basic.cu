#include <cmath>
using namespace std;
#include "GL/glew.h"
#include "GLFW/glfw3.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <curand_kernel.h>
#include <cuda_gl_interop.h>

#define TWO_PI 6.28318530718f
#define W 512
#define H 512
#define N 2048
#define MAX_STEP 8
#define MAX_DISTANCE 2.0f
#define EPSILON 1e-6f

#define block_x 32

__device__
float circleSDF(float x, float y, float cx, float cy, float r) {
	float ux = x - cx, uy = y - cy;
	return sqrtf(ux * ux + uy * uy) - r;
}

__device__
float Trace(float ox, float oy, float dx, float dy) {
	float t = 0.0f;
	for (int i = 0; i < MAX_STEP && t < MAX_DISTANCE; i++) {
		float sd = circleSDF(ox + dx * t, oy + dy * t, 0.5f, 0.5f, 0.1f);
		if (sd < EPSILON)
			return 2.0f;
		t += sd;
	}
	return 0.0f;
}

__global__
void InitRandStates(curandState *rand_states, long seed) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= W || y >= H) return;
	int offset = x + y * blockDim.x * gridDim.x;

	curand_init(seed, offset, 0, rand_states + offset);
}

__global__
void Sample(curandState *rand_states, float *buffer) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= W || y >= H) return;
	int offset = x + y * W;

	float sum = 0.0f;
	for (int i = 0; i < N; i++) {
		float a = TWO_PI * (i + curand_uniform(rand_states + offset)) / N;
		sum += Trace(float(x) / W, float(y) / H, cosf(a), sinf(a));
	}
	buffer[offset * 3 + 0] = sum / N * 255;
	buffer[offset * 3 + 1] = sum / N * 255;
	buffer[offset * 3 + 2] = sum / N * 255;
}

int main() {
	float *buffer;
	cudaMalloc(&buffer, W * H * 3 * sizeof(float));

	curandState *rand_states;
	cudaMalloc(&rand_states, W * H * sizeof(curandState));
	InitRandStates<<<dim3((W-1)/block_x+1, (H-1)/block_x+1), dim3(block_x, block_x)>>>(rand_states, time(NULL));

	Sample<<<dim3((W-1)/block_x+1, (H-1)/block_x+1), dim3(block_x, block_x)>>>(rand_states, buffer);

	float image[W * H * 3];
	cudaMemcpy(image, buffer, sizeof(image), cudaMemcpyDeviceToHost);

	uint8_t output[W * H * 3];
	for(int i = 0; i < H; ++i)
		for(int j = 0; j < W; ++j)
			for(int k = 0; k < 3; ++k) {
				output[(i * W + j) * 3 + k] = image[(i * W + j) * 3 + k];
			}
	stbi_write_png("baisc.png", W, H, 3, output, 0);
}
