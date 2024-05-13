import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from PIL import Image
from typing import Literal
import base64
import io

#*** Load the analogies dataset ***
df = pd.read_csv('data/analogies/analogies.csv')


### DATA PREPROCESSING FUNCTIONS ###    

def response_to_array(response: str, dim: int) -> np.ndarray:
    '''Converts a response string to a numpy array'''
    response = response.replace('[', '').replace(']', '').split()
    return np.array([int(x) for x in response]).reshape(dim, dim)

def task_to_arrays(task: int) -> list:
    '''Returns a list of the four matrices of a task'''
    df_task = df.loc[df['itemid'] == task]
    dim = df_task['xdim'].iloc[0]
    arrays = []
    for analogy in ['A', 'B', 'C', 'D']:
        matrix = np.array([int(char) for char in df_task[analogy].item() if char.isdigit()]).reshape(dim, dim)
        arrays.append(matrix)
    return arrays 

def task_to_input(task, input_str = ""):
    '''Returns the input string for a task'''
    task = task_to_arrays(task)

    input_str += f"Input 1: " + " ".join(str(x) for x in task[0]) + "\n"
    input_str += f"Output 1: " + " ".join(str(x) for x in task[1]) + "\n"
    input_str += f"Input 2: " + " ".join(str(x) for x in task[2]) + "\n"
    input_str += "Output 2:\n"

    return input_str


### MODEL FUNCTIONS ###

def check_response(task: int,
                   response: str) -> int:
    """
    Check the response against a task.

    Returns:
    int: 1 if the response is correct, 0 otherwise

    If the response has invalid dimensions (does not match the task dimensions), 
    it is considered incorrect.
    """
    
    # Correct answer to array
    correct_arr = task_to_arrays(task)[3]

    # Response to array 
    dim = correct_arr.shape[0]
    
    try:
        response_arr = response_to_array(response, dim)
    except ValueError:
        print('Incorrect dimensions of the response')
        return 0
        
    return 1 if np.array_equal(response_arr, correct_arr) else 0


### PLOT FUNCTIONS ###

def is_close_to_color(pixel, color, threshold):
    """Check if a pixel is within a certain threshold of a specified color."""
    return all(abs(channel - target) <= threshold for channel, target in zip(pixel, color))

def trim_whitespace(input_path, output_path, threshold=10, margin=10):
    """Trim the whitespace from an image."""
    with Image.open(input_path) as img:
        # Convert to RGB (if not already in this mode)
        img = img.convert("RGB")

        # Find the bounding box of the areas that are not close to white
        pixels = img.load()
        width, height = img.size
        bbox = None

        for y in range(height):
            for x in range(width):
                if not is_close_to_color(pixels[x, y], (255, 255, 255), threshold):
                    if bbox:
                        bbox = (min(x, bbox[0]), min(y, bbox[1]),
                                max(x, bbox[2]), max(y, bbox[3]))
                    else:
                        bbox = (x, y, x, y)

        if bbox:
            # Expand the bounding box by the specified margin
            bbox = (
                max(bbox[0] - margin, 0),
                max(bbox[1] - margin, 0),
                min(bbox[2] + margin, width),
                min(bbox[3] + margin, height)
            )

            cropped_img = img.crop(bbox)
            cropped_img.save(output_path)

def trim_whitespace_numpy(image):
    """
    Trims the whitespace from an image represented as a numpy array.
    :param image: NumPy array of the image
    :return: Cropped image
    """
    
    mask = np.any(image != [255, 255, 255, 255], axis=-1)

    # Find the bounding box of the non-white areas
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   # add 1 because slice does not include the last index

    # Crop the image
    cropped = image[x0:x1, y0:y1]
    return cropped

def encode_image(image):
    return base64.b64encode(image).decode('utf-8')

def join_images(img1_path: str, img2_path: str, save_path: str, join_direction: str) -> None:
    '''Joins two images either side by side (horizontal) or one above the other (vertical), deletes individual files, and saves the result'''

    image1 = Image.open(img1_path)
    image2 = Image.open(img2_path)

    width1, height1 = image1.size
    width2, height2 = image2.size

    if join_direction == 'horizontal':
        if height1 != height2:
            raise ValueError("Heights of the images do not match for horizontal joining.")
        total_width = width1 + width2
        max_height = max(height1, height2)
        new_image = Image.new('RGB', (total_width, max_height))
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (width1, 0))
    else:  # vertical
        if width1 != width2:
            raise ValueError("Widths of the images do not match for vertical joining.")
        total_height = height1 + height2
        max_width = max(width1, width2)
        new_image = Image.new('RGB', (max_width, total_height))
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (0, height1))

    new_image.save(save_path)

    os.remove(img1_path)
    os.remove(img2_path)


def color_recode(number: int) -> str:
    '''Recodes the color palette to match the one used in the experiment'''
    color_palette = {
        0: "black",
        1: "firebrick",
        2: "darkorange",
        3: "gold",
        4: "teal",
        5: "dodgerblue",
        6: "rebeccapurple",
        7: "hotpink",
    }
    return color_palette.get(number, "black")

def color_pallette() -> tuple:
    '''Creates a color palette for the plots'''
    color_list = [color_recode(num) for num in range(8)]
    cmap = ListedColormap(color_list)
    norm = Normalize(vmin=0, vmax=7) 
    return cmap, norm

def convert_fig_to_array(fig, dpi=300):
    '''Converts a matplotlib figure to a numpy array'''
    fig.set_dpi(dpi)
    fig.canvas.draw()
 
    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)

    # Get buffer from figure
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    # Reshape into expected shape (A, B, 4)
    expected_size = width * height * 4  # RGBA has 4 channels
    if data.size != expected_size:
        raise ValueError(f"Buffer size ({data.size}) does not match expected size ({expected_size}).")

    # Reshape and cast to 4 channels (RGBA)
    plot_array = data.reshape(height, width, 4)
    return plot_array

def task_to_base64(task):
    '''Plots the four matrices of a task'''
    plt.ioff()  # Turn off the interactive mode

    arrays = task_to_arrays(task)
    
    cmap, norm = color_pallette()

    fig, axs = plt.subplots(2, 2, figsize=(10, 5))
    axs = axs.flatten()
    for i, matrix in enumerate(arrays):
        if i == 3:
            axs[i].imshow(np.zeros((matrix.shape[0], matrix.shape[0])), cmap=cmap, norm=norm) # Black image
        else:
            axs[i].imshow(matrix, cmap=cmap, norm=norm)
        axs[i].grid(True, which='both', color='white', linewidth=0.5)
        axs[i].set_yticks([x-0.5 for x in range(1 + len(matrix))])
        axs[i].set_xticks([x-0.5 for x in range(1 + len(matrix[0]))])
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
        axs[i].tick_params(axis='both', which='both', length=0)

    # Draw arrows
    start_1 = (0.47, 0.71) 
    end_1 = (0.52, 0.71)  
    arrow_1 = plt.Arrow(start_1[0], start_1[1], end_1[0] - start_1[0], end_1[1] - start_1[1], width=0.03, color='black', transform=fig.transFigure)
    fig.add_artist(arrow_1)

    start_2 = (0.47, 0.25) 
    end_2 = (0.52, 0.25)  
    arrow_2 = plt.Arrow(start_2[0], start_2[1], end_2[0] - start_2[0], end_2[1] - start_2[1], width=0.03, color='black', transform=fig.transFigure)
    fig.add_artist(arrow_2)

    plt.tight_layout()

    # Encode figure to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return img_base64


def plot_model_responses(model: str, results: dict, arc_set: Literal['simple', 'concept'], title = '') -> None:
    '''Plots the responses of a model for all tasks in a given arc set'''

    results = results[model]

    def plot_task_response(task, response, fig, axs, row_index):
        analogy = ['a', 'b', 'c', '\nCorrect\nd', '\nGenerated\nd']

        color_list = [color_recode(num) for num in range(0, 8)]
        cmap = ListedColormap(color_list)
        norm = Normalize(vmin=0, vmax=7) 
        input_arrays = task_to_arrays(task)
        for i in range(5):
            ax = axs[row_index, i]
            if i < 4:
                matrix = input_arrays[i]
                ax.imshow(matrix, cmap=cmap, norm=norm)
                ax.set_title(analogy[i], size=30)
                if i == 0:
                    ax.set_ylabel(f"Task {task}", size = 30, rotation=0, labelpad=80, va = 'center')
            else:
                ax.imshow(response, cmap=cmap, norm=norm)
                ax.set_title(analogy[4], size=30)

            ax.grid(True, which='both', color='white', linewidth=0.5)
            ax.set_yticks([x-0.5 for x in range(1+len(matrix))])
            ax.set_xticks([x-0.5 for x in range(1+len(matrix[0]))])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='both', which='both', length=0)

    num_tasks = 8
    fig, axs = plt.subplots(num_tasks, 5, figsize=(20, num_tasks * 5))

    start, end = (1, 9) if arc_set == 'simple' else (9, 17)

    for task in range(start,end):
        dim = task_to_arrays(task)[0].shape[0]
        idx = list(range(start,end)).index(task)
        response = results[task]['response'][0]

        # If the response is not a valid array, set it to an array of zeros (black image)
        try:
            response = response_to_array(response, dim)
        except ValueError:
            response = np.zeros((dim, dim))

        plot_task_response(task, response, fig, axs, row_index=idx)

    plt.suptitle(f'{model}{title}\n\n', size=50)
    plt.tight_layout()
    plt.show()

