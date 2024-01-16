import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from LLM_data_preprocess import *
from helper_functions import *


simple_arc_df, adult_arc_df = df[df['itemid'] < 9], df[df['itemid'] > 8]
simple_arc_df.groupby('model')['correct'].sum().sort_values(ascending=False)
adult_arc_df.groupby('model')['correct'].sum().sort_values(ascending=False)



def plot_task(task: int, dpi=600) -> np.array:
    '''Plots the task as a 4x4 array'''
    arrays = task_to_arrays(task)
    analogy = ['A', 'B', 'C', 'D']

    cmap, norm = color_pallette()

    fig, axs = plt.subplots(1, 4, figsize=(4, 1))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.01, wspace=1, hspace=0.1)
    
    for i, matrix in enumerate(arrays):
        axs[i].imshow(matrix, cmap=cmap, norm=norm)
        axs[i].set_title(analogy[i], size=10)
        axs[i].grid(True, which='both', color='white', linewidth=0.5)
        axs[i].set_yticks([x-0.5 for x in range(1 + len(matrix))])
        axs[i].set_xticks([x-0.5 for x in range(1 + len(matrix[0]))])
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
        axs[i].tick_params(axis='both', which='both', length=0)

    # Draw a hyphen between the subplots
    fig.text(0.265, 0.47, '-', ha='center', va='center', fontsize=20)
    fig.text(0.735, 0.47, '-', ha='center', va='center', fontsize=20)

    plt.tight_layout()
    plot_array = convert_fig_to_array(fig, dpi=dpi)
    plt.close(fig)

    return plot_array

def get_accuracy(df: pd.DataFrame) -> tuple[np.array, np.array, np.array]:
    '''Returns the items, accuracy, and standard error of the mean of all items for a given dataframe'''
    items = df.groupby('itemid')['correct'].mean().index
    accuracy = df.groupby('itemid')['correct'].mean().values
    sem = df.groupby('itemid')['correct'].sem()
    return items, accuracy, sem

def plot_performance_errors(df: pd.DataFrame, df_incorrect: pd.DataFrame, title: str, top=3, save=False) -> None:

    items, accuracy, sem = get_accuracy(df)

    plt.rcParams.update({'font.size': 16})
    sns.set_style('ticks')

    # Create the main plot
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.subplots_adjust(left=0.38,bottom=0.15)  # Adjust left margin to create space for secondary axes
    bars = ax.barh(items, accuracy, height=0.5, color='grey')
    # ax.errorbar(accuracy, items, xerr=sem, fmt='none', c='grey', capsize=2)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='y', which='both', left=False)
    ax.set_yticks(items)
    sns.despine(offset=10, trim=False, left=True)
    ax.set_title(f'{title} Performance (n={len(df["model"].unique())})', color = 'black')
    ax.set_xlabel('Proportion Correct')

    # Add a secondary axes for annotations
    sec_ax = fig.add_axes([0.1, 0.21, 0.2, 0.62])  
    sec_ax.set_ylim(items[0], items[-1])
    for spine in sec_ax.spines.values():
        spine.set_visible(False)
    sec_ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    sec_ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    sec_ax.set_ylabel('Task')
    sec_ax.yaxis.set_label_coords(-0.15, 0.5)

    # Add annotations to the main and secondary axes
    for idx, (item, bar) in enumerate(zip(items, bars)):
        # Annotations on the main plot
        incorrect = plot_human_incorrect_responses(df_incorrect, task=item, top=top)
        incorrect = incorrect[50:500, 200:1600, :]
        #incorrect = crop_white_borders(incorrect)
        imagebox = OffsetImage(incorrect, zoom=0.06)  
        xy = (bar.get_width()+sem.iloc[idx], (bar.get_y() + bar.get_height() / 2)+0.08)
        xybox_offset = (50, 0)
        ab = AnnotationBbox(imagebox, xy, xybox=xybox_offset, boxcoords="offset points", frameon=False)
        ax.add_artist(ab)

        # Annotations on the secondary axes
        task = plot_task(item)[50:500, 200:2200, :]
        imagebox = OffsetImage(task, zoom=0.06)
        xy = (0.5, bar.get_y() + bar.get_height() / 2)  # Center in the secondary axes
        ab = AnnotationBbox(imagebox, xy, frameon=False, xycoords='data', boxcoords=("axes fraction", "data"))
        sec_ax.add_artist(ab)
    
    if save:
        fig.savefig(f'data/plots/{title}_accuracy.png', dpi=600)
    plt.show()
    plt.close()

def plot_human_incorrect_responses(df_incorrect: pd.DataFrame, task: int, top=None, dpi=600) -> np.array:

    df = df_incorrect[df_incorrect['itemid'] == task]
    if top is not None:
        df = df[:top]
    
    if top is not None:
        n_rows = top
        n_cols = 1
    else:
        n_rows = int(len(df) / 2) + (len(df) % 2)
        n_cols = 2 if len(df) > 1 else 1

    n_rows, n_cols = (1, top) if top is not None else (int(len(df) / 2) + (len(df) % 2), 2 if len(df) > 1 else 1)

        
    cmap, norm = color_pallette()
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(1 * n_cols, 1 * n_rows), dpi=dpi)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.01, wspace=1, hspace=0.1)
    axs = axs.flatten()
    
    for i, (index, row) in enumerate(df.iterrows()):
        mat = human_response_to_array(row['response'])
        ax = axs[i] if len(df) > 1 else axs  # Single subplot does not need indexing
        ax.imshow(mat, cmap=cmap, norm=norm)
        ax.set_title(f'{round(row['percent'], 1)}%', size=9)
        ax.grid(True, which='both', color='white', linewidth=0.5)
        ax.set_yticks([x-0.5 for x in range(1 + len(mat))])
        ax.set_xticks([x-0.5 for x in range(1 + len(mat[0]))])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
    
    # plt.suptitle(f'Incorrect Responses: Task {task}', size=26)
    # plt.tight_layout(rect=[0, 0, 1, 0.96])  
    
    # Handle the remaining axes if there are empty subplots
    if top is None:
        if len(df) % 2 != 0 and len(df) > 1:
            axs[-1].axis('off')  
        
    plt.tight_layout()
    
    plot_array = convert_fig_to_array(fig, dpi=dpi)
    plt.close(fig)  

    return plot_array

def get_incorrect_responses(df):
    df_incorrect = df[df['correct'] == 0]
    df_incorrect_counts = df_incorrect.groupby('itemid')['response'].value_counts(normalize=False).reset_index()
    df_incorrect_counts['percent'] = df_incorrect.groupby('itemid')['response'].value_counts(normalize=True).reset_index()['proportion']*100
    return df_incorrect_counts



def plot_responses_ind(df, model_name, dpi=120):

    df = df[df['model'] == model_name]

    cmap, norm = color_pallette()
    fig, axs = plt.subplots(8, 1, figsize=(1, 8), dpi=dpi)
    #fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.01, wspace=1, hspace=0.1)
    axs = axs.flatten()
    
    for i, (index, row) in enumerate(df.iterrows()):
        mat = human_response_to_array(row['response'])
        ax = axs[i] if len(df) > 1 else axs  # Single subplot does not need indexing
        ax.imshow(mat, cmap=cmap, norm=norm)
        ax.grid(True, which='both', color='white', linewidth=0.5)
        ax.set_yticks([x-0.5 for x in range(1 + len(mat))])
        ax.set_xticks([x-0.5 for x in range(1 + len(mat[0]))])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
    
    # Add a secondary axes for annotations
    sec_ax = fig.add_axes([0.1, 0.21, 0.2, 0.62])  
    sec_ax.set_ylim(items[0], items[-1])
    for spine in sec_ax.spines.values():
        spine.set_visible(False)
    sec_ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    sec_ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    sec_ax.set_ylabel('Task')
    sec_ax.yaxis.set_label_coords(-0.15, 0.5)

    #plt.tight_layout()
    plt.show()

plot_responses_ind(simple_arc_df, 'NousResearch/Nous-Hermes-Llama2-70b')


plot_performance_errors(simple_arc_df, get_incorrect_responses(simple_arc_df), 'Simple Arc Model', save=True)
plot_performance_errors(adult_arc_df, get_incorrect_responses(adult_arc_df), 'Adult Arc Model', save=True)





def human_response_to_array(response: str) -> np.array:
    dim = 3 if len(response) == 9 else 5
    return np.array([int(x) for x in response]).reshape(dim, dim)
