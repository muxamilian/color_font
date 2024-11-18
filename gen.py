from base import *
import torch
import json

# Sizes for each character are [text_left, text_top, text_right, text_bottom], text_sizes are [text_right - text_left, text_bottom - text_top], positions are [(img_size[0] - text_size[0]) // 2, 0]
char_images, sizes, text_sizes, positions, actual_ascii, space_width = generate_char_images(FONT_PATH)

def extract_character_from_image(img_np_array, position, text_size, size):
    # Calculate the bounds of the crop
    x, y = position
    width, height = text_size
    left, right = x, x + width
    top, bottom = y + size[1], y + size[1] + height

    # Crop the region in the PyTorch format (C, H, W)
    if len(img_np_array.shape) > 2:
        character_crop = img_np_array[:, top:bottom, left:right]
    else:
        character_crop = img_np_array[top:bottom, left:right]

    return character_crop

for i in range(len(char_images)):
    char_images[i] = extract_character_from_image(char_images[i], positions[i], text_sizes[i], sizes[i])

with open('font_info.json', 'w') as f:
    json.dump([[item.tolist() for item in char_images], sizes, text_sizes, positions, actual_ascii, space_width], f)

parsed_images = [item[:,:,0] for item in parse_tiled_image('out_images_2024-10-30T05-27-27-590906/images_42.png')]    

for i in range(len(parsed_images)):
    parsed_images[i] = extract_character_from_image(parsed_images[i], positions[i], text_sizes[i], sizes[i])

assert len(char_images) == len(parsed_images)

considering_opacity_images = []
for char_img, parsed_img in zip(char_images, parsed_images):
    considering_opacity = parsed_img * (1.-char_img)
    considering_opacity_images.append(considering_opacity)

with open('font_created.json', 'w') as f:
    json.dump([item.tolist() for item in considering_opacity_images], f)

min_color = min([np.min(item[(item != 0.) & (item != 1.)]) for item in considering_opacity_images])
max_color = max([np.max(item[(item != 0.) & (item != 1.)]) for item in considering_opacity_images])
# print(f'{min_color=}, {max_color=}')
# opacity_as_torch = torch.stack([torch.tensor(item, dtype=torch.float32) for item in considering_opacity_images], dim=0)
# save_img(considering_opacity_images, 'opacity')

def rescale(img, old_min, old_max, new_min, new_max):
    rescaled_img = np.maximum(np.minimum(new_min + ((img - old_min) * (new_max - new_min) / (old_max - old_min)), 1.0), 0.0)
    rescaled_img[img == 1.0] = 1.0
    return rescaled_img

rescaled_images = [rescale(item, min_color, max_color, 0.0, 1.0) for item in parsed_images]

for i, rescaled_img in enumerate(rescaled_images):
    assert np.min(rescaled_img) >= 0.0
    assert np.max(rescaled_img) <= 1.0

# save_img(torch.stack([torch.tensor(item, dtype=torch.float32) for item in rescaled_images], dim=0), 'rescaled')

rescaled_images = [rescale(item, min_color, max_color, 0.0, 1.0) for item in considering_opacity_images]

def mix_colors(img, color_min, color_max):
    new_img = np.tile(color_min[None, None, :], (img.shape[0],img.shape[1],1)) * (1.-img[:,:,None]) + img[:,:,None] * np.tile(color_max[None, None, :], (img.shape[0],img.shape[1],1))
    return new_img
# color_dark = [0, 0, 170] #purple
# color_bright = [254, 1, 154] #pink
color_dark = [100, 0, 0] #dark red
color_bright = [255, 0, 0] #red
# color_dark = [0, 0, 0] #black
# color_bright = [200, 200, 0] #yellow
# color_dark = [0, 60, 0] #dark green
# color_bright = [0, 160, 0] #green
# color_dark = [50, 50, 50] #dark grey
# color_bright = [140, 140, 190] #bluish
# color_dark = [255, 0, 0] #red
# color_bright = [0, 0, 255] #blue
# color_dark = [50, 50, 50] #dark grey
# color_bright = [130, 130, 130] #bright gray
# color_dark = [0, 0, 0] #black
# color_bright = [150, 150, 150] #bright gray
# color_dark = [180, 180, 0] #black
# color_bright = [220, 220, 0] #yellow
# color_dark = [80, 101, 77] #ebony
# color_bright = [156, 145, 119] #archtichoke
color_mixed = [mix_colors(item, np.array(color_dark)/255., np.array(color_bright)/255.) for item in rescaled_images]
# save_img(torch.stack([torch.tensor(np.transpose(item, (2, 0, 1)), dtype=torch.float32) for item in color_mixed], dim=0), 'mix_colors')

# With alpha
# masked = [np.concatenate((item * (1.-char[:,:,None]) + char[:,:,None], 1.-char[:,:,None]), axis=-1) for item, char in zip(color_mixed, char_images)]
# Without alpha
masked = [np.concatenate((item * (1.-char[:,:,None]) + char[:,:,None], np.ones_like(char[:,:,None])), axis=-1) for item, char in zip(color_mixed, char_images)]

# save_img(torch.stack([torch.tensor(np.transpose(item, (2, 0, 1)), dtype=torch.float32) for item in masked], dim=0), 'final')

max_height = max([item[1] for item in text_sizes])
# os.makedirs('extracted', exist_ok=True)
to_pil = T.ToPILImage()
final_extracted_char = []
for i in range(len(masked)):
    current_image = np.transpose(masked[i], (2, 0, 1))
    as_pil = to_pil(torch.tensor(current_image, dtype=torch.float32))
    final_extracted_char.append(as_pil)
    # as_pil.save(f'extracted/{i}.png')

# text = 'The quick brown fox jumps\nover the lazy dog\n0123456789'
text = 'abcdefghijklmnopqrstuvwxyz\nABCDEFGHIJKLMNOPQRSTUVWXYZ\n0123456789\n!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
# print(f'{sorted(actual_ascii)=}')
# text = actual_ascii

# Split the text into lines based on \n
lines = text.split('\n')

# Calculate the width of each line and the cumulative width positions
line_widths = []
line_cum_widths = []

for line in lines:
    sum_width = 0
    cum_widths = []
    for i in range(len(line)):
        current_char = line[i]
        if current_char == ' ':
            cum_widths.append(sum_width)
            sum_width += space_width
            continue
        char_index = actual_ascii.index(current_char)
        cum_widths.append(sum_width)
        sum_width += text_sizes[char_index][0]
    line_widths.append(sum_width)
    line_cum_widths.append(cum_widths)

# Determine the total height for the multiline text
total_height = max_height * len(lines)
# With alpha
# new_text = Image.new('RGBA', (sum_width, max_height), color=(255, 255, 255, 0))
# Without alpha
new_text = Image.new('RGBA', (max(line_widths), total_height + 8), color=(255, 255, 255, 255))

# Paste each line with centering
for line_index, line in enumerate(lines):
    line_width = line_widths[line_index]
    cum_widths = line_cum_widths[line_index]
    y_offset = line_index * max_height
    x_offset = (max(line_widths) - line_width) // 2  # Centering horizontally

    for i in range(len(line)):
        current_char = line[i]
        if current_char == ' ':
            continue
        char_index = actual_ascii.index(current_char)
        char_pos_vert = sizes[char_index][1]
        new_text.paste(final_extracted_char[char_index], 
                       (x_offset + cum_widths[i], y_offset + char_pos_vert - min([item[1] for item in sizes])))

new_text.save('text.png')
