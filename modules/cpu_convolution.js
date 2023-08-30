export {convolveCPU};

// Convoles data with kernel, storing the result in output
// Note that data is of size (dWidth * dHeight * 4), where data(x, y, c) = data(4 * (x + y * dWidth) + c).
// Note that kernel is of size (kWidth * kHeight * 4), where kernel(x, y, c) = kernel(4 * (x + y * kWidth) + c).
// The output is laid out identically with the data.
function convolveCPU(data, dWidth, dHeight, kernel, kWidth, kHeight, output) {
  // Determine kernel radii
  let rX = Math.floor((kWidth - 1) / 2);
  let rY = Math.floor((kHeight - 1) / 2);
  // Iterate over each pixel
  for(let x = 0; x < dWidth; x++) {
    for(let y = 0; y < dHeight; y++) {
      // Determine the index in the data
      let index = idx(dWidth, dHeight, x, y);
      // Create a sum placeholder for each channel
      let sums = [0.0, 0.0, 0.0, 0.0];

      // Iterate over the kernel
      for(let i = -rX; i <= rX; i++) {
        for(let j = -rY; j <= rY; j++) {
          // Determine data offset for this iteration
          let data_offset = idx(dWidth, dHeight, x - i, y - j);
          // If out of data bounds, skip this location
          // This is equivalent to a zero-padded out of bounds strategy
          if (data_offset == -1)
            continue;
          // Determine the kernel offset for this location
          let kernel_offset = idx(kWidth, kHeight, rX + i, rY + j);

          // Add product to each channel sum
          for (let c = 0; c < 4; c++) {
            sums[c] += data[data_offset + c] * kernel[kernel_offset + c];
          }
        }

      }
      // place output sums in the output
      for(let c = 0; c < 4; c++) {
        output[index + c] = sums[c];
      }
    }
  }
}

// Helper function to determine offset in an image
function idx(width, height, x, y) {
  if (x < 0 || y < 0 || x >= width || y >= height) {
    return -1;
  }
  return 4*x + 4*y*width;
}