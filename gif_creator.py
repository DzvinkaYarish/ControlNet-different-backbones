import glob
import os

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

root_path = '/gpfs/space/projects/stud_ml_22/ControlNet-different-backbones/experiments/'
experiments = [
    'laion_sd_fixed_time_1', 'laion_mlp_fixed_time_1', 'laion_conv_fixed_time_1',
    'laion_conv_hint_and_x_fixed_time', 'laion_sd_only_hint_fixed_time', 'laion_conv_decoder_fixed_time'
]
datasets = ['CC3M', 'laion', 'things']
step = 10

for experiment in experiments:
    for dataset in datasets:
        directory = os.path.join(root_path, f'{experiment}/image_log/train')

        original_files = sorted(glob.glob(os.path.join(directory, f'*{dataset}*reconstruction*')))
        condition_files = sorted(glob.glob(os.path.join(directory, f'*{dataset}*control*')))
        generated_files = sorted(glob.glob(os.path.join(directory, f'*{dataset}*samples*')))

        assert len(original_files) == len(condition_files) == len(generated_files)

        # Font settings for image number overlay
        # You can use "fc-list" to check the available fonts
        font = ImageFont.truetype('/usr/share/fonts/urw-base35/NimbusSans-Regular.otf', size=20)
        text_color = (255, 255, 255)  # White

        gif = []
        print(f'Creating {experiment}_{dataset.upper()}.gif from {len(original_files)} files ...')
        for i in tqdm(range(0, len(original_files), step)):
            original_img = Image.open(original_files[i])
            condition_img = Image.open(condition_files[i])
            generated_img = Image.open(generated_files[i])

            # Calculate the width and height of the stacked image
            width = max(original_img.width, condition_img.width, generated_img.width)
            height = original_img.height + condition_img.height + generated_img.height

            # Create a new blank image with the calculated dimensions
            stacked_image = Image.new('RGB', (width, height))

            # Paste the images onto the stacked image
            stacked_image.paste(original_img, (0, 0))
            stacked_image.paste(condition_img, (0, original_img.height))
            stacked_image.paste(generated_img, (0, original_img.height + condition_img.height))

            # Save the stacked image
            # save_path = os.path.join(directory, dataset)
            # os.mkdir(save_path, exist_ok=True)
            # stacked_image.save(os.path.join(save_path, f'stacked_{i}.png'))

            # Add image dataset overlay
            draw = ImageDraw.Draw(stacked_image)
            text = f'{dataset.upper()}'
            text_bbox = draw.textbbox((0, 0), text, font=font)
            x = 10  # Adjust the position as needed
            y = 10  # Adjust the position as needed
            draw.text((x, y), text, font=font, fill=text_color)

            # Add image number overlay
            draw = ImageDraw.Draw(stacked_image)
            text = f'Iteration {i + 1}/{len(original_files)}'
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            x = stacked_image.width - text_width - 10  # Adjust the position as needed
            y = 10  # Adjust the position as needed
            draw.text((x, y), text, font=font, fill=text_color)

            gif.append(stacked_image)

        gif[0].save(f'{experiment}_{dataset.upper()}.gif', save_all=True,
                    append_images=gif[1:], optimize=False, duration=1000, loop=0)
