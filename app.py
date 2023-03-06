import cv2
import numpy as np

# # Load source images
image1 = cv2.imread('image1.jpeg')
image2 = cv2.imread('image2.jpeg')
# image1 = cv2.imread('fusion1.jpeg')
# image2 = cv2.imread('fusion2.jpeg')

# Ensure the images have the same size and number of channels
if image1.shape != image2.shape:
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
if image1.shape[2] != image2.shape[2]:
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Multi-scale image decomposition
num_scales = 1
image1_scales = [image1]
image2_scales = [image2]
for i in range(1, num_scales):
    image1_scales.append(cv2.pyrDown(image1_scales[-1]))
    image1_scales[-1] = cv2.resize(image1_scales[-1], (image1.shape[1] // 2**(i), image1.shape[0] // 2**(i)))
    image2_scales.append(cv2.pyrDown(image2_scales[-1]))
    image2_scales[-1] = cv2.resize(image2_scales[-1], (image2.shape[1] // 2**(i), image2.shape[0] // 2**(i)))


# Base layer fusion
base_layer1 = cv2.GaussianBlur(image1, (3, 3), 0)
base_layer2 = cv2.GaussianBlur(image2, (3, 3), 0)

# Base layer fusion
base_layer_fused = cv2.addWeighted(base_layer1, 0.6, base_layer2, 0.6, 0)

# Multi-scale image fusion
synthesized_scales = []
for i in range(num_scales):
    synthesized_scales.append(cv2.addWeighted(image1_scales[i], 0.6, image2_scales[i], 0.4, 0))

# Multi-scale image reconstruction
fused_image = synthesized_scales[-1]
for i in range(num_scales-2, -1, -1):
    fused_image = cv2.pyrUp(fused_image)
    fused_image = cv2.resize(fused_image, (image1_scales[i].shape[1], image1_scales[i].shape[0]))
    synthesized_scales[i] = cv2.resize(synthesized_scales[i], (image1_scales[i].shape[1], image1_scales[i].shape[0]))
    fused_image = cv2.addWeighted(synthesized_scales[i], 0.5, fused_image, 0.5, 0)


base_layer_fused = cv2.addWeighted(base_layer1, 0.5, base_layer2, 0.5, 0)

# Detail layer fusion
detail_layers1 = [image1 - cv2.resize(cv2.pyrUp(image1_scales[i]), (image1.shape[1], image1.shape[0])) for i in range(num_scales-1)]
detail_layers2 = [image2 - cv2.resize(cv2.pyrUp(image2_scales[i]), (image2.shape[1], image2.shape[0])) for i in range(num_scales-1)]

detail_layers_fused = []
for i in range(num_scales-1):
    detail_layer_fused = cv2.addWeighted(detail_layers1[i], 0.5, detail_layers2[i], 0.5, 10)
    detail_layers_fused.append(detail_layer_fused)



# Final image fusion
fused_image = cv2.addWeighted(base_layer_fused, 0.4, fused_image, 0.4, 0)
for i in range(num_scales-1):
    fused_image += detail_layers_fused[i]

# Display the fused image
cv2.imshow('Fused Image', fused_image)
cv2.waitKey(0)
cv2.destroyAllWindows()