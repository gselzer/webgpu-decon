export function kernel_gauss(radius, sigma) {

  let length = 2 * radius + 1;
  let kernel = new Float32Array(length * length * 4)
  let s = 2 * sigma * sigma;
  let sum = 0;

  for (let x = -radius; x <= radius; x++) {
    for(let y = -radius; y <= radius; y++) {
      let idx = (x + radius) + (y + radius) * length;
      for(let c = 0; c < 4; c++) {
        kernel[4 * idx + c] = Math.exp(-(x * x + y * y) / s) / (Math.PI * s);
      }

      sum += kernel[4 * idx];
    }
  }

  for (let x = -radius; x <= radius; x++) {
    for(let y = -radius; y <= radius; y++) {
      let idx = (x + radius) + (y + radius) * length;
      for(let c = 0; c < 4; c++) {
        kernel[4 * idx + c] /= sum;
      }
    }
  }

  return kernel;
}