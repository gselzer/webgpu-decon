export {convolveGPU};

// Convoles data with kernel, storing the result in output
// Note that data is of size (dWidth * dHeight * 4), where data(x, y, c) = data(4 * (x + y * dWidth) + c).
// Note that kernel is of size (kWidth * kHeight * 4), where kernel(x, y, c) = kernel(4 * (x + y * kWidth) + c).
// The output is laid out identically with the data.
// callback is used to handle the output, when it is ready.
async function convolveGPU(data, dWidth, dHeight, kernel, kWidth, kHeight, callback) {
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
  console.log("Note the limits of this machine's device!")
  console.log(device.limits)

  // Allocate the data buffer
  let dataSize = Float32Array.BYTES_PER_ELEMENT * (data.length + 2);
  const gpuBufferFirstMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: dataSize,
    usage: GPUBufferUsage.STORAGE
  });

  // Set the data buffer equal to a Matrix struct of the data
  // First comes the matrix width and height
  // Then comes the data
  const arrayBufferFirstMatrix = gpuBufferFirstMatrix.getMappedRange();
  let data_buffer = new Float32Array(arrayBufferFirstMatrix);
  // The first entry should be the number of rows
  data_buffer[0] = dWidth;
  // The second entry should be the number of columns
  data_buffer[1] = dHeight;
  // Populate the remainder with the passed matrix
  data_buffer.set(data, 2);
  gpuBufferFirstMatrix.unmap();


  // Second Matrix

  // Allocate the kernel buffer
  let kernelSize = Float32Array.BYTES_PER_ELEMENT * (kernel.length + 2);
  const gpuBufferSecondMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: kernelSize,
    usage: GPUBufferUsage.STORAGE
  });

  // Set the kernel buffer equal to a Matrix struct of the kernel
  // First comes the matrix width and height
  // Then comes the kernel
  const arrayBufferSecondMatrix = gpuBufferSecondMatrix.getMappedRange();
  let kernel_buffer = new Float32Array(arrayBufferSecondMatrix);
  // The first entry should be the number of rows
  kernel_buffer[0] = kWidth;
  // The second entry should be the number of columns
  kernel_buffer[1] = kHeight;
  // Populate the remainder with the passed matrix
  kernel_buffer.set(kernel, 2);
  gpuBufferSecondMatrix.unmap();

  // Result Matrix
  const resultMatrixBufferSize = Float32Array.BYTES_PER_ELEMENT * (2 + 4 * dHeight * dWidth);
  const resultMatrixBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  // Set the result buffer equal to a Matrix struct of the result
  // First comes the matrix width and height
  // Then comes the result
  const arrayBufferResultMatrix = resultMatrixBuffer.getMappedRange();
  let result_buffer = new Float32Array(arrayBufferResultMatrix);
  // The first entry should be the number of rows
  result_buffer[0] = dWidth;
  // The second entry should be the number of columns
  result_buffer[1] = dHeight;
  resultMatrixBuffer.unmap();


  // Define the compute shader
  const shaderModule = device.createShaderModule({
    code: `
      struct Matrix {
        size : vec2<f32>,
        numbers: array<f32>,
      }

      @group(0) @binding(0) var<storage, read> firstMatrix : Matrix;
      @group(0) @binding(1) var<storage, read> secondMatrix : Matrix;
      @group(0) @binding(2) var<storage, read_write> resultMatrix : Matrix;

      @compute @workgroup_size(16, 16)
      fn main(
        @builtin(global_invocation_id) global_id : vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>
      ) {
        // Guard against out-of-bounds work group sizes
        if (global_id.x >= u32(firstMatrix.size.x) || global_id.y >= u32(firstMatrix.size.y)) {
          return;
        }

        // Define result matrix size
        let resultSize = vec2(i32(resultMatrix.size.x), i32(resultMatrix.size.y));
        // Define current position
        let position = vec2(i32(global_id.x), i32(global_id.y));
        // Save the kernel radius for later
        let radius = vec2(i32((secondMatrix.size.x - 1) / 2), i32((secondMatrix.size.y - 1) / 2));
        // Define current offset in data matrix
        let idx = i32(global_id.x) + i32(global_id.y) * i32(firstMatrix.size.x);
        // Perform convolution at this position
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
            // Determine data offset
            let data_offset = (i32(global_id.x) + i) + (i32(global_id.y) + j) * i32(firstMatrix.size.x);
            // Determine kernel offset
            let kernel_offset = (i + radius[0]) + ((j + radius[1]) * i32(secondMatrix.size.x));
            // Update each channel
            for (var c = 0; c < 4; c++) {
              resultMatrix.numbers[4 * i32(global_id.x + global_id.y * u32(firstMatrix.size.x)) + c] += firstMatrix.numbers[4 * data_offset + c] * secondMatrix.numbers[4 * kernel_offset + c];
            }
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

  // Bind group setup
  const bindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0 /* index */),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: gpuBufferFirstMatrix
        }
      },
      {
        binding: 1,
        resource: {
          buffer: gpuBufferSecondMatrix
        }
      },
      {
        binding: 2,
        resource: {
          buffer: resultMatrixBuffer
        }
      }
    ]
  });
  

  // Commands submission
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  const workgroupCountX = Math.ceil(dWidth / 16);
  const workgroupCountY = Math.ceil(dHeight / 16);
  console.log(`Dispatching (${workgroupCountX} x ${workgroupCountY}) workgroups`)
  passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
  passEncoder.end();

  // Get a GPU buffer for reading in an unmapped state.
  const gpuReadBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  // Encode commands for copying buffer to buffer.
  commandEncoder.copyBufferToBuffer(
    resultMatrixBuffer /* source buffer */,
    0 /* source offset */,
    gpuReadBuffer /* destination buffer */,
    0 /* destination offset */,
    resultMatrixBufferSize /* size */
  );

  // Submit GPU commands.
  const gpuCommands = commandEncoder.finish();
  var startTime = performance.now()
  device.queue.submit([gpuCommands]);


  // Read buffer.
  await gpuReadBuffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = gpuReadBuffer.getMappedRange();
  var endTime = performance.now()
  console.log(`GPU computation took ${endTime - startTime} milliseconds`)
  // Pass the output onto the callback.
  callback(new Float32Array(arrayBuffer).slice(2));
};