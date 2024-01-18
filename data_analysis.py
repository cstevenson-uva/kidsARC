import pandas as pd
import os
from helper_functions import *
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
from typing import Optional

### Helper functions
def get_accuracy(df: pd.DataFrame) -> tuple[np.array, np.array, np.array]:
    '''Returns the items, accuracy, and standard error of the mean of all items for a given dataframe'''
    items = df.groupby('itemid')['correct'].mean().index
    accuracy = df.groupby('itemid')['correct'].mean().values
    sem = df.groupby('itemid')['correct'].sem()
    return items, accuracy, sem

def response_to_array(response: str) -> np.array:
    dim = 3 if len(response) == 9 else 5
    return np.array([int(x) for x in response]).reshape(dim, dim)

def get_incorrect_responses(df: pd.DataFrame) -> pd.DataFrame:
    df_incorrect = df[df['correct'] == 0]
    df_incorrect_counts = df_incorrect.groupby('itemid')['response'].value_counts(normalize=False).reset_index()
    df_incorrect_counts['percent'] = df_incorrect.groupby('itemid')['response'].value_counts(normalize=True).reset_index()['proportion']*100
    return df_incorrect_counts

### Plotting functions

def plot_task(task: int, show: bool = False, dpi: int = 600) -> Optional[np.array]:
    '''
    Plots the task as a 1x4 array
    If show is False, the plot is returned as a numpy array instead of shown
    '''
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

    if not show:
        plot_array = convert_fig_to_array(fig, dpi=dpi)
        plt.close(fig)

        return plot_array
    else:
        plt.show()
        plt.close(fig)

def plot_incorrect_responses(df_incorrect: pd.DataFrame, task: int, top=None, dpi=600) -> np.array:

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
        mat = response_to_array(row['response'])
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

def plot_performance_errors(df: pd.DataFrame, df_incorrect: pd.DataFrame, title: str, top=3, save=False) -> None:

    items, accuracy, sem = get_accuracy(df)

    plt.rcParams.update({'font.size': 16})
    sns.set_style('ticks')

    # Create the main plot
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.subplots_adjust(left=0.38,bottom=0.15)  # Adjust left margin to create space for secondary axes
    bars = ax.barh(items, accuracy, height=0.5, color='grey')
    ax.errorbar(accuracy, items, xerr=sem, fmt='none', c='grey', capsize=2)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='y', which='both', left=False)
    ax.set_yticks(items)
    sns.despine(offset=10, trim=False, left=True)
    ax.set_title(f'{title} Performance (n={len(df["participant_fk"].unique())})', color = 'black')
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
        incorrect = plot_incorrect_responses(df_incorrect, task=item, top=top)
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



get_incorrect_responses(df_human).reset_index().groupby('itemid')['response'].count()
get_incorrect_responses(df_models).reset_index()




#TODO: Filter out responses all but two items 
    
df_models = pd.read_csv('data/model/models.csv')
df_human = pd.read_csv('data/human/kidsarc.csv')

# RT to seconds
df_human['rt_s'] = df_human['rt'] / 1000

# Split into kids and concept ARC
df_human_kids, df_human_concept = df_human[df_human['itemid'] < 9], df_human[df_human['itemid'] > 8]
df_models_kids, df_models_concept = df_models[df_models['itemid'] < 9], df_models[df_models['itemid'] > 8]


# filter out participants with less than 6 responses
no_response = df_human_kids[df_human_kids['response'] == '0'*9]
to_remove = no_response.groupby('participant_fk').filter(lambda x: len(x) > 6)['participant_fk'].unique()
df_human_kids = df_human_kids[~df_human_kids['participant_fk'].isin(to_remove)]

no_response = df_human_concept[df_human_concept['response'] == '0'*25]
to_remove = no_response.groupby('participant_fk').filter(lambda x: len(x) > 6)['participant_fk'].unique()
df_human_concept = df_human_concept[~df_human_concept['participant_fk'].isin(to_remove)]

print(len(df_human_kids.groupby('participant_fk')))
print(len(df_human_concept.groupby('participant_fk')))


### Incorrect responses
kids_arc_incorrect, concept_arc_incorrect = get_incorrect_responses(df_human_kids), get_incorrect_responses(df_human_concept)

plot_performance_errors(df_human_kids, kids_arc_incorrect, 'Kids ARC', save=True)
plot_performance_errors(df_human_concept, concept_arc_incorrect, 'Concept ARC', save=True)






df = df_human_kids[df_human_kids['age'] > 8]

plot_performance_errors(df, get_incorrect_responses(df), 'Kids ARC Age', save=True)

df = df_human_concept[df_human_concept['age'] > 20]
plot_performance_errors(df, get_incorrect_responses(df), 'Kids Concept ARC', save=True)




###  Day validation ###
kids_arc_df_sat = df_human_kids[df_human_kids['saveitem_timestamp'].str.startswith('2023-12-09')]
kids_arc_df_sun = df_human_kids[df_human_kids['saveitem_timestamp'].str.startswith('2023-12-10')]
concept_arc_df_sat = df_human_concept[df_human_concept['saveitem_timestamp'].str.startswith('2023-12-09')]
concept_arc_df_sun = df_human_concept[df_human_concept['saveitem_timestamp'].str.startswith('2023-12-10')]
dfs_day_valid = [kids_arc_df_sat, kids_arc_df_sun, concept_arc_df_sat, concept_arc_df_sun]


# ERROR PLOTS
titles = ['Kids ARC (Saturday)', 'Kids ARC (Sunday)', 'Concept ARC (Saturday)', 'Concept ARC (Sunday)']

for df, title in zip(dfs_day_valid, titles):
    df_incorrect = get_incorrect_responses(df)
    plot_performance_errors(df, df_incorrect, title, save=True)

join_images('data/plots/Kids ARC (Saturday)_accuracy.png', 
            'data/plots/Kids ARC (Sunday)_accuracy.png', 
            save_path='data/plots/Kids ARC_accuracy_validation.png')
join_images('data/plots/Concept ARC (Saturday)_accuracy.png',
            'data/plots/Concept ARC (Sunday)_accuracy.png',
            save_path='data/plots/Concept ARC_accuracy_validation.png')


# PLOT ACCURACY BY TASK
acc_sat = np.append(get_accuracy(kids_arc_df_sat)[1], get_accuracy(concept_arc_df_sat)[1])
acc_sun = np.append(get_accuracy(kids_arc_df_sun)[1], get_accuracy(concept_arc_df_sun)[1])
tasks = list(range(1,17))

threshold = 0.1 
width = 0.35  
x = np.arange(len(acc_sat))

fig, ax = plt.subplots(figsize=(10, 5))
for i in range(len(acc_sat)):
    diff = abs(acc_sat[i] - acc_sun[i])
    alpha = 1.0 if diff >= threshold else 0.5  # Increase opacity for small differences
    bar1 = ax.bar(x[i] - width/2, acc_sat[i], width, label='Sat', align='center', color='teal', alpha=alpha)
    bar2 = ax.bar(x[i] + width/2, acc_sun[i], width, label='Sun', align='center', color='indianred', alpha=alpha)

ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.set_title('Performance Day Validation')
ax.set_xlabel('Task')
ax.set_ylabel('Performance')
ax.set_ylim(0, 1)

sat_patch = mpatches.Patch(color='teal', alpha=1.0, label='Sat')
sun_patch = mpatches.Patch(color='indianred', alpha=1.0, label='Sun')
ax.legend(handles=[sat_patch, sun_patch])

plt.show()


# PLOT RESPONSE TIME BY TASK
sat_RT = concept_arc_df_sat.groupby('itemid')['rt_s'].mean().to_numpy()
sun_RT = concept_arc_df_sun.groupby('itemid')['rt_s'].mean().to_numpy()
tasks = list(range(9,17))

threshold = 5
width = 0.35  
x = np.arange(len(sat_RT))
fig, ax = plt.subplots(figsize=(8, 5))
for i in range(len(sat_RT)):
    diff = abs(sat_RT[i] - sun_RT[i])
    alpha = 1.0 if diff >= threshold else 0.5  # Increase opacity for small differences
    bar1 = ax.bar(x[i] - width/2, sat_RT[i], width, label='Sat', align='center', color='teal', alpha=alpha)
    bar2 = ax.bar(x[i] + width/2, sun_RT[i], width, label='Sun', align='center', color='indianred', alpha=alpha)
ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.set_title('RT Day Validation')
ax.set_xlabel('Task')
ax.set_ylabel('RT (s)')
sat_patch = mpatches.Patch(color='teal', alpha=1.0, label='Sat')
sun_patch = mpatches.Patch(color='indianred', alpha=1.0, label='Sun')
ax.legend(handles=[sat_patch, sun_patch])
plt.show()












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
        ax.set_title(f'{row['count']} ({round(row['percent'], 1)}%)', size=9)
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
    plt.show()
    plt.close(fig)  

inc=get_incorrect_responses(df_nondup)
plot_human_incorrect_responses(inc, 13, top=None)