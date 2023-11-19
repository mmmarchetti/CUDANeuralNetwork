# CUDA Neural Network Digit Recognition

This project implements a neural network for digit recognition using CUDA, NVIDIA's parallel computing platform. The neural network is implemented in C++ with CUDA extensions for parallel processing on compatible NVIDIA GPUs.

## Overview

The neural network consists of one hidden layer and uses the sigmoid activation function. The CUDA kernels are utilized for matrix multiplication and element-wise sigmoid activation to accelerate the training and prediction processes on the GPU.

## Files

- **neural_network.cu**: The main CUDA C++ source file containing the neural network implementation.
  
## Usage

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
   ```

2. Compile the code using NVCC:

   ```bash
   nvcc -o your_executable_name neural_network.cu -lcudart
   ```

3. Run the executable:

   ```bash
   ./your_executable_name
   ```

## Dependencies

- NVIDIA GPU with CUDA support
- CUDA Toolkit

## Notes

- This is a simple example for educational purposes and may need further modifications for real-world applications.
- Ensure that your GPU supports CUDA, and the CUDA Toolkit is properly installed on your system.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
