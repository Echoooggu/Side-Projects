# Poisson Image Editing

## Descriptions
Ten cases, each with a provided source image, a target image, and potentially a mask image. If there is no mask, then draw a mask by mouse through a popped-out window.  
Cannot manipulate where to blend the image in the target. The python files use mixed-gradient blending, and therefore only suit some types of images.

## Algorithms（*mixed*-gradient blending)
**Seamless image fusion with mixed gradients (preserving dominant structures from either source or target). A functional optimization problem that seeks a function whose gradient closely approximates the maximum of the gradient of the source image and that of the target image.**  
1. Obtain max(gradient of the source image, gradient of the target image)  
2. Obtain a functional minimization problem, whose solution is exactly what we want.  
3. The functional minimization problem can be turned into a [poisson equation](https://en.wikipedia.org/wiki/Poisson%27s_equation) (divergence of gradient = Laplacian) by using [variational methods](https://en.wikipedia.org/wiki/Variational_method_(quantum_mechanics)).  
4. The poisson equation is also a linear equation of the desired function, i.e. the solution of the functional optimization problem.  
5. Discretize the Laplacian operator (e.g., 5-point stencil) to form a sparse linear system.
6. Solve the system using NumPy/SciPy, with the divergence of the mixed gradient field as the right-hand side.
7. For color images, process each channel independently.

ps. Mixed-gradient blending is especially useful for images containing holes like [case4], nearly transparent images like [case5], and when inserting one object close to another, but not for cases like [case8]. For classic poisson image editing, simply change the maximum of the two gradients to only the gradient of the source image, with the rest of the steps the same.

## Potential problems
### 1. The mask region is mostly black in the result, with blurred edges.
A: uint8 should be float64
### 2. The source is blurred within the mask region in the result.
A: Should probably use MIXED gradient for poisson blending.
### 3. The mask region seems to be in the middle (or any other position) of the source, but it turns out that the ultimate position of processed source image is not and even seems arbitrary.
A: The source image and the target image should be processed to align with each other.

## Acknowledgements
https://github.com/willemmanuel/poisson-image-editing.git  
[2003 poisson image editing paper](https://github.com/Echoooggu/CV_selfLearn/blob/main/poisson_image_editing/2003_poisson_image_editing.pdf)
