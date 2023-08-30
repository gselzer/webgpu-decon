import { deconvolveCPU } from "../modules/cpu_deconvolution.js"

// input data: 3 x 3 x 4
let data = new Float32Array(36);
for(let i = 0; i < data.length; i++) {
    data[i] = (255 / 9);
}

let kernel = new Float32Array(36);
for(let i = 0; i < kernel.length; i++) {
    kernel[i] = 1 / 9;
}

let output = new Float32Array(36);

deconvolveCPU(data, 3, 3, kernel, 3, 3, output);

for(let i = 0; i < output.length; i++) {
    if (i >= 16 && i < 20){
        if (Math.abs(output[i] - 255) > 1) {
            throw new Error("Deconvolution doesn't work")
        }
    }
    else {
        if (Math.abs(output[i] - 0) > 0.1) {
            throw new Error("Deconvolution doesn't work")
        }
    }
}
