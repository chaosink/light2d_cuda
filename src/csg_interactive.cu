#include <cmath>
#include <vector>
using namespace std;
#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include <curand_kernel.h>
#include <cuda_gl_interop.h>

#include "FPS.hpp"

#define TWO_PI 6.28318530718f
#define W 512
#define H 512
#define N 16
#define MAX_STEP 64
#define MAX_DISTANCE 2.0f
#define EPSILON 1e-6f

#define block_x 32

bool updated = false;
int op = 0;
__device__ int op_dev;
float light[4] = {0.4f, 0.5f, 0.2f, 1.f};
__device__ float light_dev[4];

struct Result {
	float sd;
	glm::vec3 emission;
};

static void errorCallback(int error, const char* description) {
	fprintf(stderr, "Error: %s\n", description);
}

static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if(glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
		op = (op + 1) % 5;
		updated = true;
	}
	if(glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
		op = 1 - 1;
		updated = true;
	}
	if(glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
		op = 2 - 1;
		updated = true;
	}
	if(glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
		op = 3 - 1;
		updated = true;
	}
	if(glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) {
		op = 4 - 1;
		updated = true;
	}
	if(glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) {
		op = 5 - 1;
		updated = true;
	}
	if(glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS) {
		op = 6 - 1;
		updated = true;
	}
	if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
		light[0] += 0.01f;
		updated = true;
	}
	if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
		light[1] += 0.01f;
		updated = true;
	}
	if(glfwGetKey( window, GLFW_KEY_A) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
		light[0] -= 0.01f;
		updated = true;
	}
	if(glfwGetKey( window, GLFW_KEY_S) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
		light[1] -= 0.01f;
		updated = true;
	}
	if(glfwGetKey( window, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS) {
		light[2] += 0.01f;
		updated = true;
	}
	if(glfwGetKey( window, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS) {
		light[2] -= 0.01f;
		updated = true;
	}
	if(glfwGetKey( window, GLFW_KEY_EQUAL) == GLFW_PRESS) {
		light[3] += 0.1f;
		updated = true;
	}
	if(glfwGetKey( window, GLFW_KEY_MINUS) == GLFW_PRESS) {
		light[3] -= 0.1f;
		updated = true;
	}

}

__device__
Result UnionOp(Result a, Result b) {
	return a.sd < b.sd ? a : b;
}

__device__
Result IntersectOp(Result a, Result b) {
	Result r = a.sd < b.sd ? a : b;
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
Result Scene(float x, float y) {
	Result a = {
		CircleSDF(x, y, light_dev[0], light_dev[1], light_dev[2]),
		{light_dev[3], light_dev[3], light_dev[3]}
	};
	Result b = {
		CircleSDF(x, y, 0.7f, 0.5f, 0.15f),
		{1.f, 0.1f, 0.5f}
	};
	Result c = {
		CircleSDF(x, y, 0.5f, 0.2f, 0.04f),
		{0.1f, 0.2f, 0.3f}
	};
	Result d = {
		CircleSDF(x, y, 0.2f, 0.8f, 0.06f),
		{0.3f, 0.9f, 0.2f}
	};
	Result result = UnionOp(a, b);
	switch(op_dev) {
		case 0: result = UnionOp(a, b); break;
		case 1: result = IntersectOp(a, b); break;
		case 2: result = SubtractOp(a, b); break;
		case 3: result = SubtractOp(b, a); break;
		case 4: result = ComplementOp(a); break;
	}
	result = UnionOp(result, c);
	result = UnionOp(result, d);
	return result;
}

__device__
glm::vec3 Trace(float ox, float oy, float dx, float dy) {
	float t = 0.001f;
	for (int i = 0; i < MAX_STEP && t < MAX_DISTANCE; i++) {
		Result r = Scene(ox + dx * t, oy + dy * t);
		if (r.sd < EPSILON)
			return r.emission;
		t += r.sd;
	}
	return {0.f, 0.f, 0.f};
}

__global__
void Sample(curandState *rand_states, float *buffer, int c_sample) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= W || y >= H) return;
	int offset = x + y * W;

	glm::vec3 sum;
	for (int i = 0; i < N; i++) {
		float a = TWO_PI * (i + curand_uniform(rand_states + offset)) / N;
		sum += Trace(float(x) / W, float(y) / H, cosf(a), sinf(a));
	}
	sum /= float(N);
	buffer[offset * 3 + 0] = (buffer[offset * 3 + 0] * (c_sample - 1) + sum.r) / c_sample;
	buffer[offset * 3 + 1] = (buffer[offset * 3 + 1] * (c_sample - 1) + sum.g) / c_sample;
	buffer[offset * 3 + 2] = (buffer[offset * 3 + 2] * (c_sample - 1) + sum.b) / c_sample;
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
	vector<float> image(W * H * 3);

	GLFWwindow* window;

	glfwSetErrorCallback(errorCallback);
	if(!glfwInit()) exit(EXIT_FAILURE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	// glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
	window = glfwCreateWindow(W, H, "csg_interactive", NULL, NULL);
	if(!window) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwSetKeyCallback(window, KeyCallback);
	glfwMakeContextCurrent(window);
	if(glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}
	glfwSwapInterval(1);

	GLuint buffer;
	glGenBuffers(1, &buffer);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(float) * image.size(), image.data(), GL_DYNAMIC_DRAW);
	// glRasterPos2f(-1, 1);
	// glPixelZoom(1, -1);

	cudaGraphicsResource *resource;
	cudaGraphicsGLRegisterBuffer(&resource, buffer, cudaGraphicsMapFlagsNone);
	float *dev_ptr;
	size_t size;

	curandState *rand_states;
	cudaMalloc(&rand_states, W * H * sizeof(curandState));
	InitRandStates<<<dim3((W-1)/block_x+1, (H-1)/block_x+1), dim3(block_x, block_x)>>>(rand_states, time(NULL));

	cudaGraphicsMapResources(1, &resource, NULL);
	cudaGraphicsResourceGetMappedPointer((void**)&dev_ptr, &size, resource);

	FPS fps;
	int c_sample = 0;
	while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && !glfwWindowShouldClose(window)) {
		if(updated) {
			glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(float) * image.size(), image.data(), GL_DYNAMIC_DRAW);
			updated = false;
			c_sample = 0;
		}

		c_sample++;
		cudaGraphicsMapResources(1, &resource, NULL);
		cudaMemcpyToSymbol(op_dev, &op, sizeof(int));
		cudaMemcpyToSymbol(light_dev, light, sizeof(light));
		Sample<<<dim3((W-1)/block_x+1, (H-1)/block_x+1), dim3(block_x, block_x)>>>(rand_states, dev_ptr, c_sample);
		cudaGraphicsUnmapResources(1, &resource, NULL);

		glClear(GL_COLOR_BUFFER_BIT);
		glDrawPixels(W, H, GL_RGB, GL_FLOAT, 0);
		glfwSwapBuffers(window);
		glfwPollEvents();

		fps.Update();
	}
}
