#include <cmath>
using namespace std;
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
using namespace cv;
#include <curand_kernel.h>
#include <cuda_gl_interop.h>

#define TWO_PI 6.28318530718f
#define W 512
#define H 512
#define N 64
#define MAX_STEP 64
#define MAX_DISTANCE 2.0f
#define EPSILON 1e-6f

#define block_x 32

struct Result {
	float sd, emissive;
};

__device__
Result unionOp(Result a, Result b) {
	return a.sd < b.sd ? a : b;
}

__device__
Result intersectOp(Result a, Result b) {
	Result r = a.sd > b.sd ? b : a;
	r.sd = a.sd > b.sd ? a.sd : b.sd;
	return r;
}

__device__
Result subtractOp(Result a, Result b) {
	Result r = a;
	r.sd = (a.sd > -b.sd) ? a.sd : -b.sd;
	return r;
}

__device__
Result complementOp(Result a) {
	a.sd = -a.sd;
	return a;
}

__device__
float CircleSDF(float x, float y, float cx, float cy, float r) {
	float ux = x - cx, uy = y - cy;
	return sqrtf(ux * ux + uy * uy) - r;
}

__device__
Result Scene(float x, float y) {
#if 0
    Result r1 = { CircleSDF(x, y, 0.3f, 0.3f, 0.10f), 2.0f };
    Result r2 = { CircleSDF(x, y, 0.3f, 0.7f, 0.05f), 0.8f };
    Result r3 = { CircleSDF(x, y, 0.7f, 0.5f, 0.10f), 0.0f };
    return unionOp(unionOp(r1, r2), r3);
#else
    Result a = { CircleSDF(x, y, 0.4f, 0.5f, 0.20f), 1.0f };
    Result b = { CircleSDF(x, y, 0.6f, 0.5f, 0.20f), 0.8f };
    return unionOp(a, b);
    // return intersectOp(a, b);
    // return subtractOp(a, b);
    // return subtractOp(b, a);
#endif
}

__device__
float Trace(float ox, float oy, float dx, float dy) {
	float t = 0.001f;
	for (int i = 0; i < MAX_STEP && t < MAX_DISTANCE; i++) {
		Result r = Scene(ox + dx * t, oy + dy * t);
		if (r.sd < EPSILON)
			return r.emissive;
		t += r.sd;
	}
	return 0.f;
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
		sum += Trace(float(x) / W, float(y) / H, cos(a), sin(a));
	}
	buffer[offset * 3 + 0] = sum / N * 255;
	buffer[offset * 3 + 1] = sum / N * 255;
	buffer[offset * 3 + 2] = sum / N * 255;
}

__global__
void InitRandStates(curandState *rand_states, long seed) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= W || y >= H) return;
	int offset = x + y * blockDim.x * gridDim.x;

	curand_init(seed, offset, 0, rand_states + offset);
}

int main() {
	float *buffer;
	cudaMalloc(&buffer, W * H * 3 * sizeof(float));

	curandState *rand_states;
	cudaMalloc(&rand_states, W * H * sizeof(curandState));
	InitRandStates<<<dim3((W-1)/block_x+1, (H-1)/block_x+1), dim3(block_x, block_x)>>>(rand_states, time(NULL));

	Sample<<<dim3((W-1)/block_x+1, (H-1)/block_x+1), dim3(block_x, block_x)>>>(rand_states, buffer);

	Mat img = Mat(H, W, CV_32FC3);
	cudaMemcpy(img.data, buffer, W * H * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	imwrite("csg.png", img);
}
