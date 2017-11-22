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
#define N 2048
#define MAX_STEP 64
#define MAX_DISTANCE 2.0f
#define EPSILON 1e-6f

#define block_x 32

struct Result {
	float sd, emissive;
};

__device__
Result UnionOp(Result a, Result b) {
	return a.sd < b.sd ? a : b;
}

__device__
Result IntersectOp(Result a, Result b) {
	Result r = a.sd > b.sd ? b : a;
	r.sd = a.sd > b.sd ? a.sd : b.sd;
	return r;
}

__device__
Result SubtractOp(Result a, Result b) {
	Result r = a;
	r.sd = (a.sd > -b.sd) ? a.sd : -b.sd;
	return r;
}

__device__
Result ComplementOp(Result a) {
	a.sd = -a.sd;
	return a;
}

__device__
float CircleSDF(float x, float y, float cx, float cy, float r) {
	float ux = x - cx, uy = y - cy;
	return sqrtf(ux * ux + uy * uy) - r;
}

__device__
float PlaneSDF(float x, float y, float px, float py, float nx, float ny) {
	return (x - px) * nx + (y - py) * ny;
}

__device__
float SegmentSDF(float x, float y, float ax, float ay, float bx, float by) {
	float vx = x - ax, vy = y - ay, ux = bx - ax, uy = by - ay;
	float t = fmaxf(fminf((vx * ux + vy * uy) / (ux * ux + uy * uy), 1.0f), 0.0f);
	float dx = vx - ux * t, dy = vy - uy * t;
	return sqrtf(dx * dx + dy * dy);
}

__device__
float CapsuleSDF(float x, float y, float ax, float ay, float bx, float by, float r) {
	return SegmentSDF(x, y, ax, ay, bx, by) - r;
}

__device__
float BoxSDF(float x, float y, float cx, float cy, float theta, float sx, float sy) {
	float costheta = cosf(theta), sintheta = sinf(theta);
	float dx = fabs((x - cx) * costheta + (y - cy) * sintheta) - sx;
	float dy = fabs((y - cy) * costheta - (x - cx) * sintheta) - sy;
	float ax = fmaxf(dx, 0.0f), ay = fmaxf(dy, 0.0f);
	return fminf(fmaxf(dx, dy), 0.0f) + sqrtf(ax * ax + ay * ay);
}

__device__
float TriangleSDF(float x, float y, float ax, float ay, float bx, float by, float cx, float cy) {
	float d = fminf(fminf(
		SegmentSDF(x, y, ax, ay, bx, by),
		SegmentSDF(x, y, bx, by, cx, cy)),
		SegmentSDF(x, y, cx, cy, ax, ay));
	return
		(bx - ax) * (y - ay) > (by - ay) * (x - ax) &&
		(cx - bx) * (y - by) > (cy - by) * (x - bx) &&
		(ax - cx) * (y - cy) > (ay - cy) * (x - cx) ? -d : d;
}

__device__
Result Scene(float x, float y) {
	Result a = {  CircleSDF(x, y, 0.2f, 0.2f, 0.1f), 1.0f};
	Result b = {   PlaneSDF(x, y, 0.0f, 0.5f, 0.0f, 1.0f), 0.8f};
	Result c = { CapsuleSDF(x, y, 0.15f, 0.85f, 0.4f, 0.8f, 0.1f), 1.0f};
	Result d = {     BoxSDF(x, y, 0.8f, 0.3f, TWO_PI / 5.0f, 0.25f, 0.1f), 1.0f};
	Result e = {     BoxSDF(x, y, 0.5f, 0.5f, TWO_PI / 16.0f, 0.3f, 0.1f) - 0.1f, 1.0f};
	Result f = {TriangleSDF(x, y, 0.5f, 0.2f, 0.8f, 0.8f, 0.3f, 0.6f), 1.0f};
	Result g = {TriangleSDF(x, y, 0.5f, 0.2f, 0.8f, 0.8f, 0.3f, 0.6f) - 0.1f, 1.0f};
	Result result = a;
	result = UnionOp(result, c);
	result = UnionOp(result, d);
	result = UnionOp(result, f);
	// return a;
	// return b;
	// return c;
	// return d;
	// return e;
	// return f;
	// return g;
	return result;
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
		sum += Trace(float(x) / W, float(y) / H, cosf(a), sinf(a));
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
	imwrite("shapes.png", img);
}
