import { convolveCPU } from "./modules/cpu_convolution.js"
import { convolveGPU } from "./modules/gpu_convolution.js"
import { deconvolveCPU } from "./modules/cpu_deconvolution.js"
import { deconvolveGPU } from "./modules/gpu_deconvolution.js"
import { kernel_gauss } from "./modules/kernels.js"

/* DOM setup */

// Grab the input button
const fileinput = document.getElementById('fileinput')

// Grab the canvas
const image_canvas = document.getElementById('image_canvas')

// Grab the four buttons
const convolve_cpu = document.getElementById('convolve_cpu')
const deconvolve_cpu = document.getElementById('deconvolve_cpu')
const convolve_gpu = document.getElementById('convolve_gpu')
const deconvolve_gpu = document.getElementById('deconvolve_gpu')

// Get the 2d (as opposed to "3d") drawing context on the canvas, returns CanvasRenderingContext2D
const image_ctx = image_canvas.getContext('2d')


/* Variables setup */

// Similar to document.createElement('img') except we don't need it on the document
const srcImage = new Image()
let imgData = null;


/* DOM functions */

// When the user selects an image, load in the data
fileinput.onchange = function (e) {
  // Check validity
  if (e.target.files && e.target.files.item(0)) {
    // Set the src of the new Image()
    srcImage.src = URL.createObjectURL(e.target.files[0])
  }
}

// When the image loads, propagate the image through the state.
srcImage.onload = function () {
  // Copy the image's dimensions to the canvas, which will show the preview of the edits
  image_canvas.width = srcImage.width
  image_canvas.height = srcImage.height

  // draw the image at with no offset (0,0) and with the same dimensions as the image
  image_ctx.drawImage(srcImage, 0, 0, srcImage.width, srcImage.height)

  // Get an ImageData object representing the underlying pixel data for the area of the canvas
  imgData = image_ctx.getImageData(0, 0, srcImage.width, srcImage.height)
}

// Transfers the changes we made to be displayed on the canvas
function commitChanges(data) {
  if (imgData == null) return
  if (imgData.data.length != data.length) {
    console.log("The passed data is not the same length as the original!")
  }
  
  // Copy over the current pixel changes to the image
  for (let i = 0; i < imgData.data.length; i++) {
    imgData.data[i] = data[i]
  }

  // Update the 2d rendering canvas with the image we just updated so the user can see
  image_ctx.putImageData(imgData, 0, 0, 0, 0, srcImage.width, srcImage.height)
}

/* Buttons */

deconvolve_cpu.onclick = function() {
  // Setup data
  let data = imgData.data.slice();
  let dWidth = imgData.width;
  let dHeight = imgData.height;
  // Setup kernel
  let radius = 4, sigma = 1;
  let kernel = kernel_gauss(radius, sigma);
  let kSideLength = 2 * radius + 1;
  // Setup output
  let output = new Float32Array(data.length);

  // Time CPU decon
  var startTime = performance.now()
  deconvolveCPU(data, dWidth, dHeight, kernel, kSideLength, kSideLength, output);
  var endTime = performance.now()
  // Log time taken
  console.log(`CPU deconvolution took ${endTime - startTime} milliseconds`)
  // Update display
  commitChanges(output);
}

convolve_cpu.onclick = function() {
  // Setup data
  let data = imgData.data.slice();
  let dWidth = imgData.width;
  let dHeight = imgData.height;
  // Setup kernel
  let radius = 4;
  let kernel = kernel_gauss(radius, 1)
  let kSideLength = 2 * radius + 1;
  //Setup output
  let output = new Float32Array(data.length)

  // Time CPU convolution
  var startTime = performance.now()
  convolveCPU(data, dWidth, dHeight, kernel, kSideLength, kSideLength, output);
  var endTime = performance.now()
  // Log time taken
  console.log(`CPU convolution took ${endTime - startTime} milliseconds`)
  // Update display
  commitChanges(output);
}

convolve_gpu.onclick = function() {
  // Setup data
  let data = Float32Array.from(imgData.data.slice());
  // Setup kernel
  let radius = 4;
  let currentKernel = kernel_gauss(radius, 1)

  // Call GPU Convolution
  // NB: I couldn't figure out how to get the output of the GPU convolution here.
  // So I deal with it inside of the function.
  convolveGPU(data, imgData.width, imgData.height, currentKernel, 2 * radius + 1, 2 * radius + 1, commitChanges);
}

deconvolve_gpu.onclick = function() {
  // Setup data
  let image = Float32Array.from(imgData.data.slice());
  // Setup kernel
  let radius = 4;
  let currentKernel = kernel_gauss(radius, 1)

  // Call GPU Decon
  // NB: I couldn't figure out how to get the output of the GPU deconvolution here.
  // So I deal with it inside of the function.
  deconvolveGPU(image, imgData.width, imgData.height, currentKernel, 2 * radius + 1, 2 * radius + 1, commitChanges);
}

