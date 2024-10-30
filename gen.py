from base import *
import torch

# Sizes for each character are [text_left, text_top, text_right, text_bottom], text_sizes are [text_right - text_left, text_bottom - text_top], positions are [(img_size[0] - text_size[0]) // 2, 0]
char_images, sizes, text_sizes, positions, actual_ascii, space_width = generate_char_images(FONT_PATH)
parsed_images = [item[:,:,0] for item in parse_tiled_image('out_images_2024-10-30T05-27-27-590906/images_42.png')]    

assert len(char_images) == len(parsed_images)

considering_opacity_images_binary = []
considering_opacity_images = []
for char_img, parsed_img in zip(char_images, parsed_images):
    considering_opacity = (parsed_img / (1.-char_img))
    cleaned_considering_opacity = np.nan_to_num(considering_opacity, nan=1.0, posinf=1.0, neginf=1.0)
    considering_opacity_images.append(cleaned_considering_opacity)
    good_pixels = char_img == 0.0
    considering_opacity = np.ones_like(char_img)
    considering_opacity[good_pixels] = parsed_img[good_pixels]
    considering_opacity_images_binary.append(considering_opacity)


min_color = min([np.min(item[(item != 0.) & (item != 1.)]) for item in considering_opacity_images_binary])
max_color = max([np.max(item[(item != 0.) & (item != 1.)]) for item in considering_opacity_images_binary])
print(f'{min_color=}, {max_color=}')
opacity_as_torch = torch.stack([torch.tensor(item, dtype=torch.float32) for item in considering_opacity_images_binary], dim=0)
# save_img(opacity_as_torch, 'opacity_binary')
opacity_as_torch = torch.stack([torch.tensor(item, dtype=torch.float32) for item in considering_opacity_images], dim=0)
# save_img(opacity_as_torch, 'opacity')

def rescale(img, old_min, old_max, new_min, new_max):
    rescaled_img = np.maximum(np.minimum(new_min + ((img - old_min) * (new_max - new_min) / (old_max - old_min)), 1.0), 0.0)
    rescaled_img[img == 1.0] = 1.0
    return rescaled_img

rescaled_images = [rescale(item, min_color, max_color, 0.0, 1.0) for item in parsed_images]

for i, rescaled_img in enumerate(rescaled_images):
    assert np.min(rescaled_img) >= 0.0
    assert np.max(rescaled_img) <= 1.0

# save_img(torch.stack([torch.tensor(item, dtype=torch.float32) for item in rescaled_images], dim=0), 'rescaled')

rescaled_images = [rescale(item, min_color, max_color, 0.0, 1.0) for item in considering_opacity_images_binary]

def mix_colors(img, color_min, color_max):
    new_img = np.tile(color_min[None, None, :], (224,224,1)) * (1.-img[:,:,None]) + img[:,:,None] * np.tile(color_max[None, None, :], (224,224,1))
    return new_img
# color_dark = [0, 0, 170] #purple
# color_bright = [254, 1, 154] #pink
# color_dark = [100, 0, 0] #dark red
# color_bright = [255, 0, 0] #red
# color_dark = [0, 0, 0] #black
# color_bright = [200, 200, 0] #yellow
# color_dark = [0, 60, 0] #dark green
# color_bright = [0, 160, 0] #green
color_dark = [50, 50, 50] #dark grey
color_bright = [140, 140, 190] #bluish
# color_dark = [255, 0, 0] #red
# color_bright = [0, 0, 255] #blue
# color_dark = [50, 50, 50] #dark grey
# color_bright = [130, 130, 130] #bright gray
# color_dark = [0, 0, 0] #black
# color_bright = [150, 150, 150] #bright gray
color_mixed = [mix_colors(item, np.array(color_dark)/255., np.array(color_bright)/255.) for item in rescaled_images]
# save_img(torch.stack([torch.tensor(np.transpose(item, (2, 0, 1)), dtype=torch.float32) for item in color_mixed], dim=0), 'mix_colors')

masked = [item * (1.-char[:,:,None]) + char[:,:,None] for item, char in zip(color_mixed, char_images)]

# save_img(torch.stack([torch.tensor(np.transpose(item, (2, 0, 1)), dtype=torch.float32) for item in masked], dim=0), 'final')

def extract_character_from_image(img_np_array, position, text_size, size):
    # Calculate the bounds of the crop
    x, y = position
    width, height = text_size
    left, right = x, x + width
    top, bottom = y + size[1], y + size[1] + height

    # Crop the region in the PyTorch format (C, H, W)
    character_crop = img_np_array[:, top:bottom, left:right]

    return character_crop

os.makedirs('extracted', exist_ok=True)
to_pil = T.ToPILImage()
final_extracted_char = []
for i in range(len(masked)):
    current_image = np.transpose(masked[i], (2, 0, 1))
    crop = extract_character_from_image(current_image, positions[i], text_sizes[i], sizes[i])
    as_pytorch = to_pil(torch.tensor(crop, dtype=torch.float32))
    final_extracted_char.append(as_pytorch)
    # as_pytorch.save(f'extracted/{i}.png')

# max_width = max([item[0] for item in text_sizes])
max_height = max([item[1] for item in text_sizes])

text = 'Gabi & Max forever!!!;'
# text = actual_ascii

sum_width = 0
cum_widths = []
for i in range(len(text)):
    current_char = text[i]
    if current_char == ' ':
        cum_widths.append(sum_width)
        sum_width += space_width
        continue
    char_index = actual_ascii.index(current_char)
    cum_widths.append(sum_width)
    sum_width += text_sizes[char_index][0]

new_text = Image.new('RGB', (sum_width, max_height), color=(255, 255, 255))

for i in range(len(text)):
    current_char = text[i]
    if current_char == ' ':
        continue
    char_index = actual_ascii.index(current_char)
    char_pos_vert = sizes[char_index][1]
    new_text.paste(final_extracted_char[char_index], (cum_widths[i], char_pos_vert - min([item[1] for item in sizes])))

new_text.save('text.png')

# TODO: font with not-constant width, different color combinations
