from PIL import Image, ImageChops

# Open the two images
img1 = Image.open('out_images/images_0.png')
img2 = Image.open('out_images/images_10.png')

# Compute the difference
diff = ImageChops.difference(img1, img2)

# Save the difference image
diff.save('diff.png')

# If the images are exactly the same, diff will be an empty image
if not diff.getbbox():
    print("The images are identical.")
else:
    print("The images have differences.")
