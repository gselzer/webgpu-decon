import { convolveCPU } from "../modules/cpu_convolution.js"

// input data: 3 x 3 x 4
let data = new Float32Array(36);
for(let i = 0; i < 4; i++) {
    data[16+i] = 255;
}

let kernel = new Float32Array(36);
for(let i = 0; i < 36; i++) {
    kernel[i] = 1 / 9;
}

let output = new Float32Array(36);

convolveCPU(data, 3, 3, kernel, 3, 3, output)

for(let i = 0; i < output.length; i++) {
    if (Math.abs(output[i] - (255 / 9)) > 1e-6) {
        throw new Error("Convolution is wrong");
    }
}
