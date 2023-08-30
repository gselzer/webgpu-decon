import { convolveCPU } from "./cpu_convolution.js";

export {deconvolveCPU};

// Deconvoles data with kernel, storing the result in output.
// Deconvolution is based on the Richardson-Lucy algorithm, running 30 iterations.
// See https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution for more information.
// Note that data is of size (dWidth * dHeight * 4), where data(x, y, c) = data(4 * (x + y * dWidth) + c).
// Note that kernel is of size (kWidth * kHeight * 4), where kernel(x, y, c) = kernel(4 * (x + y * kWidth) + c).
// The output is laid out identically with the data.
function deconvolveCPU(data, dWidth, dHeight, kernel, kWidth, kHeight, output) {

  // The starting output should be equal to the input
  for(let i = 0; i < data.length; i++) {
    output[i] = data[i];
  }

  // Create some data buffers
  let est_convolved = new Float32Array(data.length);
  let relative_blur = new Float32Array(data.length);
  let error_est = new Float32Array(data.length);

  // Compute the "flipped" kernel for later.
  let kernel_hat = kernel.reverse()

  // Iterate
  let maxIterations = 30
  for(let i = 0; i < maxIterations; i++){

    // Convolve the output of the previous iteration with the kernel
    convolveCPU(output, dWidth, dHeight, kernel, kWidth, kHeight, est_convolved);
    // Divide the original image by the product of that convolution
    for(let j = 0; j < relative_blur.length; j++) {
      relative_blur[j] = data[j] / est_convolved[j];
    }
    // Then convolve THAT with the flipped kernel
    convolveCPU(relative_blur, dWidth, dHeight, kernel_hat, kWidth, kHeight, error_est);
    // Set the output to THAT times the output of the second convolution
    for(let j = 0; j < output.length; j++) {
      output[j] = output[j] * error_est[j];
    }
  }

}