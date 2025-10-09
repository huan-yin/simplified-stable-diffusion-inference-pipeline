from PIL import Image
import numpy as np

def calculate_image_difference(image1_path, image2_path):
    """
    Calculate the pixel-wise difference between two images
    
    Parameters:
    image1_path: Path to the first image
    image2_path: Path to the second image

    
    Returns:
    diff_array: Numpy array of difference values
    total_diff: Total difference value
    diff_pixel_count: Number of differing pixels
    total_pixel_count: Total number of pixels in the image
    """
    # Read images and convert to RGB mode (to handle images of different modes uniformly)
    img1 = Image.open(image1_path).convert('RGB')
    img2 = Image.open(image2_path).convert('RGB')

    # Check if the sizes of the two images are the same
    if img1.size != img2.size:
        raise ValueError(f"The sizes of the two images are different, so pixel-wise comparison is not possible. "
                         f"Image 1 size: {img1.size}, Image 2 size: {img2.size}")
    
    # Convert images to numpy arrays for calculation
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    
    # Calculate pixel-wise difference (take absolute value)
    diff_array = np.abs(img1_array - img2_array)
    
    # Calculate difference statistics
    total_diff = np.sum(diff_array)
    total_pixel_count = img1_array.shape[0] * img1_array.shape[1]
    # Determine if each pixel has a difference (a pixel is considered differing if any channel has a difference)
    diff_pixel_count = np.sum(np.any(diff_array > 0, axis=2))
    
    return diff_array, total_diff, diff_pixel_count, total_pixel_count

if __name__ == "__main__":
    image1_path = "images/sd15_cat_with_pipline.png"
    image2_path = "images/sd15_cat_without_pipline.png"
    diff_array, total_diff, diff_pixel_count, total_pixel_count = calculate_image_difference(image1_path, image2_path)
    
    print(f"Total image difference value: {total_diff}")
    print(f"Number of differing pixels: {diff_pixel_count}")
    print(f"Total number of image pixels: {total_pixel_count}")
    print(f"Proportion of differing pixels: {diff_pixel_count / total_pixel_count:.2%}")