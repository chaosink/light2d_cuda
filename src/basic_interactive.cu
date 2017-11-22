#include <cmath>
using namespace std;
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
using namespace cv;
#include <curand_kernel.h>
#include <cuda_gl_interop.h>

#include "fps.hpp"

#define TWO_PI 6.28318530718f
#define W 512
#define H 512
#define N 16
#define MAX_STEP 10
#define MAX_DISTANCE 2.0f
#define EPSILON 1e-6f

#define block_x 32

bool updated = false;
float light[2] = {0.5f, 0.5f};
__device__ float light_dev[2];

static void error_callback(int error, const char* description) {
	fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
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
}

__device__
float circleSDF(float x, float y, float cx, float cy, float r) {
	float ux = x - cx, uy = y - cy;
	return sqrtf(ux * ux + uy * uy) - r;
}

__device__
float trace(float ox, float oy, float dx, float dy) {
	float t = 0.0f;
	for (int i = 0; i < MAX_STEP && t < MAX_DISTANCE; i++) {
		float sd = circleSDF(ox + dx * t, oy + dy * t, light_dev[0], light_dev[1], 0.1f);
		if (sd < EPSILON)
			return 2.0f;
		t += sd;
	}
	return 0.0f;
}

__global__
void Sample(curandState *rand_states, float *buffer, int c_sample) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= W || y >= H) return;
	int offset = x + y * W;

	float sum = 0.0f;
	for (int i = 0; i < N; i++) {
		// float a = TWO_PI * rand() / RAND_MAX;
		// float a = TWO_PI * i / N;
		float a = TWO_PI * (i + curand_uniform(rand_states + offset)) / N;
		sum += trace(float(x) / W, float(y) / H, cos(a), sin(a));
	}
	buffer[offset * 3 + 0] = (buffer[offset * 3 + 0] * (c_sample - 1) + sum / N) / c_sample;
	buffer[offset * 3 + 1] = (buffer[offset * 3 + 1] * (c_sample - 1) + sum / N) / c_sample;
	buffer[offset * 3 + 2] = (buffer[offset * 3 + 2] * (c_sample - 1) + sum / N) / c_sample;
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
	Mat image(H, W, CV_32FC3); // init buffer, all 0

	GLFWwindow* window;

	glfwSetErrorCallback(error_callback);
	if(!glfwInit()) exit(EXIT_FAILURE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
	window = glfwCreateWindow(W, H, "Simple example", NULL, NULL);
	if(!window) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwSetKeyCallback(window, key_callback);
	glfwMakeContextCurrent(window);
	if(glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}
	glfwSwapInterval(1);

	GLuint buffer;
	glGenBuffers(1, &buffer);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, W * H * 3 * sizeof(float), image.data, GL_DYNAMIC_DRAW);
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
			glBufferData(GL_PIXEL_UNPACK_BUFFER, W * H * 3 * sizeof(float), image.data, GL_DYNAMIC_DRAW);
			updated = false;
			c_sample = 0;
		}

		c_sample++;
		cudaGraphicsMapResources(1, &resource, NULL);
		cudaMemcpyToSymbol(light_dev, light, sizeof(float) * 2);
		Sample<<<dim3((W-1)/block_x+1, (H-1)/block_x+1), dim3(block_x, block_x)>>>(rand_states, dev_ptr, c_sample);
		cudaGraphicsUnmapResources(1, &resource, NULL);

		glClear(GL_COLOR_BUFFER_BIT);
		glDrawPixels(W, H, GL_RGB, GL_FLOAT, 0);
		glfwSwapBuffers(window);
		glfwPollEvents();

		fps.Update();
	}
}
