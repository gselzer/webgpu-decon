export {deconvolveGPU};

// Deconvoles data with kernel, storing the result in output.
// Deconvolution is based on the Richardson-Lucy algorithm, running 30 iterations.
// See https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution for more information.
// Note that data is of size (dWidth * dHeight * 4), where data(x, y, c) = data(4 * (x + y * dWidth) + c).
// Note that kernel is of size (kWidth * kHeight * 4), where kernel(x, y, c) = kernel(4 * (x + y * kWidth) + c).
// The output is laid out identically with the data.
// callback is used to handle the output, when it is ready.
async function deconvolveGPU(data, dWidth, dHeight, kernel, kWidth, kHeight, callback) {
  // Ensure WebGPU is supported
  if (!("gpu" in navigator)) {
    console.log("WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag.");
    return;
  }

  // Get the device
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    console.log("Failed to get GPU adapter.");
    return;
  }
  const device = await adapter.requestDevice();
  // Since Decon is so much more complicated,
  // Let's also display the device limits
  console.log("Note the limits of this machine's device!")
  console.log(device.limits)

  // Create data buffers
  const dataBufferSize = Float32Array.BYTES_PER_ELEMENT * (2 + data.length);
  const gpuBufferReferenceMatrix = dataBuffer(device, data, dWidth, dHeight);
  const gpuBufferResultMatrix = resultBuffer(device, data, dWidth, dHeight);
  const gpuBufferIntermediate1 = dataBuffer(device, data, dWidth, dHeight);
  const gpuBufferIntermediate2 = dataBuffer(device, data, dWidth, dHeight);

  // Create kernel buffers
  const gpuBufferKernelMatrix = dataBuffer(device, kernel, kWidth, kHeight);
  const gpuBufferKernelStarMatrix = dataBuffer(device, kernel, kWidth, kHeight);

  // Since I cannot figure out how to pass through all of the data at once, 
  // we perform the computation in 4 steps. Since there can be only one kernel
  // in a WebGPU pipeline, we use case logic within the kernel. That case logic
  // is dictated by these buffers.
  // Step 1: We convolve the current iteration with the kernel
  const gpuBufferParamsStep1 = paramsBuffer(device, [1]);
  // Step 2: We divide the original data by the result of Step 1
  const gpuBufferParamsStep2 = paramsBuffer(device, [2]);
  // Step 3: We convolve step 2 with the "flipped kernel"
  const gpuBufferParamsStep3 = paramsBuffer(device, [3]);
  // Step 4: We multiply the current iteration by Step 3 to produce the next iteration.
  const gpuBufferParamsStep4 = paramsBuffer(device, [4]);

  // Compute shader code
  const shaderModule = device.createShaderModule({
    code: `
      struct Matrix {
        size : vec2<f32>,
        numbers: array<f32>,
      }

      @group(0) @binding(0) var<storage, read> data : Matrix;
      @group(0) @binding(1) var<storage, read> secondMatrix : Matrix;
      @group(0) @binding(2) var<storage, read_write> resultMatrix : Matrix;
      @group(0) @binding(3) var<storage, read> params : array<i32>;

      @compute @workgroup_size(16, 16)
      fn main(
        @builtin(global_invocation_id) global_id : vec3<u32>,
      ) {
        // Guard against out-of-bounds work group sizes
        if (global_id.x >= u32(data.size.x) || global_id.y >= u32(data.size.y)) {
          return;
        }
        // Determine position in the matrix
        let idx = i32(global_id.x + global_id.y * u32(data.size.x));
        let resultSize = vec2(i32(resultMatrix.size.x), i32(resultMatrix.size.y));

        let mode = params[0];
        // Steps 1 and 3 require convolution
        if (mode == 1 || mode == 3) {
          for (var c = 0; c < 4; c++) {
            resultMatrix.numbers[4 * idx + c] = 0;
          }
          let position = vec2(i32(global_id.x), i32(global_id.y));
          let radius = vec2(i32((secondMatrix.size.x - 1) / 2), i32((secondMatrix.size.y - 1) / 2));
          for (var i = -radius[0]; i <= radius[0]; i++) {
            for (var j = -radius[1]; j <= radius[1]; j++) {
              // pass over entries on the left or top
              if (i > position.x || j > position.y) {
                continue;
              }
              // pass over entries on the right or bottom
              if (i + position.x > resultSize.x || j + position.y > resultSize.y) {
                continue;
              }
              let data_offset = (i32(global_id.x) + i) + (i32(global_id.y) + j) * i32(data.size.x);
              let kernel_offset = (i + radius[0]) + ((j + radius[1]) * i32(secondMatrix.size.x));
              for (var c = 0; c < 4; c++) {
                resultMatrix.numbers[4 * idx + c] += data.numbers[4 * data_offset + c] * secondMatrix.numbers[4 * kernel_offset + c];
              }
            }
          }
        }
        // Step 2 requires division
        else if (mode == 2) {
          for (var c = 0; c < 4; c++) {
            resultMatrix.numbers[4 * idx + c] = data.numbers[4 * idx + c] / secondMatrix.numbers[4 * idx + c];
          }
        }
        // Step 4 requires multiplication
        else if (mode == 4) {
          for (var c = 0; c < 4; c++) {
            resultMatrix.numbers[4 * idx + c] *= data.numbers[4 * idx + c];
          }
        }
      }
    `
  });
  
  // Pipeline setup
  const computePipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: shaderModule,
      entryPoint: "main"
    }
  });

  // Bind group layouts

  // Step 1
  const bindGroup1 = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0 /* index */),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: gpuBufferResultMatrix
        }
      },
      {
        binding: 1,
        resource: {
          buffer: gpuBufferKernelMatrix
        }
      },
      {
        binding: 2,
        resource: {
          buffer: gpuBufferIntermediate1,
        }
      },
      {
        binding: 3,
        resource: {
          buffer: gpuBufferParamsStep1,
        }
      }
    ]
  });

  // Step 2
  const bindGroup2 = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0 /* index */),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: gpuBufferReferenceMatrix
        }
      },
      {
        binding: 1,
        resource: {
          buffer: gpuBufferIntermediate1
        }
      },
      {
        binding: 2,
        resource: {
          buffer: gpuBufferIntermediate2,
        }
      },
      {
        binding: 3,
        resource: {
          buffer: gpuBufferParamsStep2
        }
      },
    ]
  });

  // Step 3
  const bindGroup3 = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0 /* index */),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: gpuBufferIntermediate2
        }
      },
      {
        binding: 1,
        resource: {
          buffer: gpuBufferKernelStarMatrix
        }
      },
      {
        binding: 2,
        resource: {
          buffer: gpuBufferIntermediate1,
        }
      },
      {
        binding: 3,
        resource: {
          buffer: gpuBufferParamsStep3
        }
      },
    ]
  });

  // Step 4
  const bindGroup4 = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0 /* index */),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: gpuBufferIntermediate1
        }
      },
      {
        binding: 1,
        resource: {
          buffer: gpuBufferKernelMatrix
        }
      },
      {
        binding: 2,
        resource: {
          buffer: gpuBufferResultMatrix,
        }
      },
      {
        binding: 3,
        resource: {
          buffer: gpuBufferParamsStep4
        }
      },
    ]
  });

  // Commands submission
  const workgroupCountX = Math.ceil(dWidth / 16);
  const workgroupCountY = Math.ceil(dHeight / 16);
  console.log(`Dispatching (${workgroupCountX} x ${workgroupCountY}) workgroups`)

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  // Encode 30 dispatches of each of the 4 steps, in order.
  for(let i = 0; i < 30; i++) {
    // Step 1
    passEncoder.setBindGroup(0, bindGroup1);
    passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    // Step 2
    passEncoder.setBindGroup(0, bindGroup2);
    passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    // Step 3
    passEncoder.setBindGroup(0, bindGroup3);
    passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    // Step 4
    passEncoder.setBindGroup(0, bindGroup4);
    passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
  }
  passEncoder.end();

  // Get a GPU buffer for reading in an unmapped state.
  const gpuReadBuffer = device.createBuffer({
    size: dataBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  // Encode commands for copying buffer to buffer.
  commandEncoder.copyBufferToBuffer(
    gpuBufferResultMatrix /* source buffer */,
    0 /* source offset */,
    gpuReadBuffer /* destination buffer */,
    0 /* destination offset */,
    dataBufferSize /* size */
  );

  // Submit GPU commands.
  const gpuCommands = commandEncoder.finish();
  var startTime = performance.now()
  device.queue.submit([gpuCommands]);


  // Read buffer.
  await gpuReadBuffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = gpuReadBuffer.getMappedRange();
  var endTime = performance.now()
  console.log(`GPU deconvolution took ${endTime - startTime} milliseconds`)
  callback(new Float32Array(arrayBuffer).slice(2));
};

// Helper function to create a data buffer
function dataBuffer(device, data, width, height) {
  // Allocate the data buffer
  let dataSize = Float32Array.BYTES_PER_ELEMENT * (data.length + 2);
  const gpuBufferFirstMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: dataSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const arrayBufferFirstMatrix = gpuBufferFirstMatrix.getMappedRange();

  let data_buffer = new Float32Array(arrayBufferFirstMatrix);
  // The first entry should be the number of rows
  data_buffer[0] = width;
  // The second entry should be the number of columns
  data_buffer[1] = height;
  // Populate the remainder with the passed matrix
  data_buffer.set(data, 2);
  gpuBufferFirstMatrix.unmap();

  return gpuBufferFirstMatrix;
}

// Helper function to create a result buffer
function resultBuffer(device, data, width, height) {
  // Allocate the data buffer
  let dataSize = Float32Array.BYTES_PER_ELEMENT * (4 * width * height + 2);
  const gpuBufferFirstMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: dataSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const arrayBufferFirstMatrix = gpuBufferFirstMatrix.getMappedRange();

  let data_buffer = new Float32Array(arrayBufferFirstMatrix);
  // The first entry should be the number of rows
  data_buffer[0] = width;
  // The second entry should be the number of columns
  data_buffer[1] = height;
  // Populate the remainder with the passed matrix
  data_buffer.set(data, 2);
  gpuBufferFirstMatrix.unmap();

  return gpuBufferFirstMatrix;
}

// Helper function to create a parameter buffer
function paramsBuffer(device, params) {
  // Allocate the data buffer
  let dataSize = Int32Array.BYTES_PER_ELEMENT * (params.length);
  const gpuBufferFirstMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: dataSize,
    usage: GPUBufferUsage.STORAGE
  });
  const arrayBufferFirstMatrix = gpuBufferFirstMatrix.getMappedRange();

  let data_buffer = new Int8Array(arrayBufferFirstMatrix);
  for (let i = 0; i < params.length; i++){
    data_buffer[i] = params[i];
  }
  gpuBufferFirstMatrix.unmap();

  return gpuBufferFirstMatrix;
}