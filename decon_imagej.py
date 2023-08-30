#@ OpService ops
#@ UIService ui
#@ ImgPlus img
#@OUTPUT ImgPlus convolved
#@OUTPUT ImgPlus deconvolved

from java.lang import System

# parameters
sigma = 1
numIterations = 30

# convert to float (TODO: make sure deconvolution op works on other types)
img_float=ops.convert().float32(img)

# create and show the gaussian kernel
if img_float.numDimensions()==3:
	psf=ops.create().kernelGauss([sigma, sigma, 0])
elif img_float.numDimensions()==2:
	psf=ops.create().kernelGauss([sigma, sigma])

# Convolve
start = System.currentTimeMillis()
convolved = ops.filter().convolve(img_float, psf)
end = System.currentTimeMillis()
print("Convolution execution time: " + str(end - start) + " ms")

# deconvolve
start = System.currentTimeMillis()
deconvolved=ops.deconvolve().richardsonLucyTV(convolved, psf, numIterations, 0.01)
end = System.currentTimeMillis()
print("Deconvolution execution time: " + str(end - start) + " ms")