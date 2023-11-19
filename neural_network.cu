#include <iostream>
#include <cmath>
#include <vector>

// CUDA kernel for matrix multiplication
__global__ void matrixMultiply(float *a, float *b, float *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

// Sigmoid activation function
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// CUDA kernel for element-wise sigmoid activation
__global__ void sigmoidActivation(float *a, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] = sigmoid(a[idx]);
    }
}

// Neural network class
class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize);
    ~NeuralNetwork();
    void train(std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& target, int epochs);
    void predict(std::vector<float>& input, std::vector<float>& output);

private:
    int inputSize;
    int hiddenSize;
    int outputSize;

    float *d_input;
    float *d_hiddenWeights;
    float *d_hiddenBiases;
    float *d_hiddenOutput;
    float *d_outputWeights;
    float *d_outputBiases;
    float *d_output;

    void initializeWeights();
    void forwardPass(float *inputData);
    void backwardPass(float *targetData);
};

NeuralNetwork::NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
    : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize) {

    // Allocate device memory
    cudaMalloc((void**)&d_input, inputSize * sizeof(float));
    cudaMalloc((void**)&d_hiddenWeights, inputSize * hiddenSize * sizeof(float));
    cudaMalloc((void**)&d_hiddenBiases, hiddenSize * sizeof(float));
    cudaMalloc((void**)&d_hiddenOutput, hiddenSize * sizeof(float));
    cudaMalloc((void**)&d_outputWeights, hiddenSize * outputSize * sizeof(float));
    cudaMalloc((void**)&d_outputBiases, outputSize * sizeof(float));
    cudaMalloc((void**)&d_output, outputSize * sizeof(float));

    // Initialize weights and biases
    initializeWeights();
}

NeuralNetwork::~NeuralNetwork() {
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_hiddenWeights);
    cudaFree(d_hiddenBiases);
    cudaFree(d_hiddenOutput);
    cudaFree(d_outputWeights);
    cudaFree(d_outputBiases);
    cudaFree(d_output);
}

void NeuralNetwork::initializeWeights() {
    // Initialize weights and biases with small random values
    // You may need a more sophisticated initialization method in practice
    // This is a simple example
    std::vector<float> h_hiddenWeights(inputSize * hiddenSize);
    std::vector<float> h_outputWeights(hiddenSize * outputSize);
    std::vector<float> h_hiddenBiases(hiddenSize);
    std::vector<float> h_outputBiases(outputSize);

    for (int i = 0; i < inputSize * hiddenSize; ++i) {
        h_hiddenWeights[i] = 0.01f * rand() / RAND_MAX;
    }

    for (int i = 0; i < hiddenSize * outputSize; ++i) {
        h_outputWeights[i] = 0.01f * rand() / RAND_MAX;
    }

    for (int i = 0; i < hiddenSize; ++i) {
        h_hiddenBiases[i] = 0.01f * rand() / RAND_MAX;
    }

    for (int i = 0; i < outputSize; ++i) {
        h_outputBiases[i] = 0.01f * rand() / RAND_MAX;
    }

    // Copy data to device
    cudaMemcpy(d_hiddenWeights, h_hiddenWeights.data(), inputSize * hiddenSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputWeights, h_outputWeights.data(), hiddenSize * outputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hiddenBiases, h_hiddenBiases.data(), hiddenSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputBiases, h_outputBiases.data(), outputSize * sizeof(float), cudaMemcpyHostToDevice);
}

void NeuralNetwork::forwardPass(float *inputData) {
    // Copy input data to device
    cudaMemcpy(d_input, inputData, inputSize * sizeof(float), cudaMemcpyHostToDevice);

    // Forward pass
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((hiddenSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (1 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Compute hidden layer output
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_input, d_hiddenWeights, d_hiddenOutput, 1, inputSize, hiddenSize);

    // Add biases to hidden layer output
    cudaMemcpy(d_hiddenOutput, d_hiddenBiases, hiddenSize * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    // Apply sigmoid activation to hidden layer output
    sigmoidActivation<<<(hiddenSize + threadsPerBlock.x - 1) / threadsPerBlock.x, threadsPerBlock>>>(d_hiddenOutput, hiddenSize);

    // Compute final output
    matrixMultiply<<<1, outputSize>>>(d_hiddenOutput, d_outputWeights, d_output, 1, hiddenSize, outputSize);

    // Add biases to final output
    cudaMemcpy(d_output, d_outputBiases, outputSize * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    // Apply sigmoid activation to final output
    sigmoidActivation<<<(outputSize + threadsPerBlock.x - 1) / threadsPerBlock.x, threadsPerBlock>>>(d_output, outputSize);
}

void NeuralNetwork::backwardPass(float *targetData) {
    // Compute output layer error
    cudaMemcpy(d_output, targetData, outputSize * sizeof(float), cudaMemcpyHostToDevice);

    // Compute output layer gradients
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((outputSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (1 + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_output, d_hiddenOutput, d_outputWeights, outputSize, 1, hiddenSize);

    // Compute hidden layer error
    matrixMultiply<<<1, hiddenSize>>>(d_output, d_outputWeights, d_hiddenOutput, 1, outputSize, hiddenSize);

    // Compute hidden layer gradients
    matrix
