import pandas as pd
from helper_functions import *
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from typing import Optional
from matplotlib.ticker import FuncFormatter
Image.MAX_IMAGE_PIXELS = None


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

def get_unique_responses(df: pd.DataFrame) -> pd.DataFrame:
    df_responses = df.groupby('itemid')['response'].value_counts(normalize=False).reset_index()
    df_responses['percent'] = df.groupby('itemid')['response'].value_counts(normalize=True).reset_index()['proportion']*100
    df_responses = df_responses[~df_responses['response'].astype(str).str.match('^0+$')] # Remove empty responses
    return df_responses


### Plotting functions
def plot_task(task: int, title: str, show: bool = False, dpi: int = 600) -> Optional[np.array]:
    '''
    Plots the task as a 1x4 array
    If show is False, the plot is returned as a numpy array instead of shown
    '''
    arrays = task_to_arrays(task)
    analogy = ['A', 'B', 'C', 'D']

    cmap, norm = color_pallette()

    fig, axs = plt.subplots(2, 2, figsize=(2, 1))
    axs = axs.flatten()
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.01, wspace=1, hspace=0.1)
    
    for i, matrix in enumerate(arrays):
        if i != 3:
            axs[i].imshow(matrix, cmap=cmap, norm=norm)
            #axs[i].set_title(analogy[i], size=10)
            axs[i].grid(True, which='both', color='white', linewidth=0.5)
            axs[i].set_yticks([x-0.5 for x in range(1 + len(matrix))])
            axs[i].set_xticks([x-0.5 for x in range(1 + len(matrix[0]))])
            axs[i].set_xticklabels([])
            axs[i].set_yticklabels([])
            axs[i].tick_params(axis='both', which='both', length=0)
        else:
            axs[i].axis('off')  # Remove the last subplot

    fig.subplots_adjust(wspace=0.1)  # Adjust the space between subplots if necessary
    arrow_properties = dict(arrowstyle="->", color="black", lw=1)

    axs[0].annotate('', xy=(2.4, 0.5), xytext=(1.4, 0.5),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=arrow_properties)
    axs[2].annotate('', xy=(2.4, 0.5), xytext=(1.4, 0.5),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=arrow_properties)
    
    # add text to the left
    bbox1 = axs[0].get_position()
    bbox2 = axs[1].get_position()
    center_y = bbox1.y0 - 0.05  # Adjust this as needed for spacing

    # Add text at the calculated position
    fig.text(0.05, center_y+0.01, f'Item {title}', ha='center', va='center', size=16, rotation=90)

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
        #ax.set_title(f'{row['count']} ({round(row['percent'], 1)}%)', size=9)
        # Define title text and position
        title_text = f"{round(row['percent'])}%"
        # Position the title to the right of the matrix
        ax.text(1.1, 0.5, title_text, transform=ax.transAxes, size=10)
        ax.grid(True, which='both', color='white', linewidth=0.5)
        ax.set_yticks([x-0.5 for x in range(1 + len(mat))])
        ax.set_xticks([x-0.5 for x in range(1 + len(mat[0]))])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
        # if row['response'] in possible_responses_dict[task]:
        #     # Set the colors of the axes spines
        #     for spine in ax.spines.values():
        #         spine.set_edgecolor('green')
        #         spine.set_linewidth(4)  # Increase the edge size here

    
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


### Get the data ready
df_models = pd.read_csv('data/model/models.csv')
df_human = pd.read_csv('data/human/kidsarc.csv')
df_models, df_human = df_models.rename(columns={'model': 'respondent_id'}), df_human.rename(columns={'participant_fk': 'respondent_id'})
df_models['respondent_id'] = df_models['respondent_id'].apply(lambda x: x.split('/')[1] if '/' in x else x)  # Remove the 'zero-one-ai/' prefix
df_models = df_models[~(df_models['respondent_id'] == 'gpt-4-vision_image')] # Remove the gpt-4 vision with image only prompt
df_models['respondent_id'] = df_models['respondent_id'].apply(lambda x: 'gpt-4-vision' if x == 'gpt-4-vision_image_text' else x)
items = df_human[['itemid', 'A', 'B', 'C', 'D']].drop_duplicates().set_index('itemid')
df_human['rt_s'] = df_human['rt'] / 1000 # RT to seconds


#### Correct the correct column
def correct_correct(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Corrects the 'correct' column based on the response and the correct answer.
    '''
    # Function to check if the response is correct
    def check_correct(row):
        return int(row['response'] in possible_responses_dict[row['itemid']])

    # Apply the check for each column
    df['correct'] = df.apply(check_correct, axis=1)

    return df

# Identified possible correct responses for each task
possible_responses_dict = {
    1: ['000000100'],
    2: ['770000770'],
    3: ['000666000'],
    4: ['000404404'],
    5: ['777700000', '770700700'],
    6: ['070070033'],
    7: ['102102102', '000112112'],
    8: ['000333444'],
    9: ['0110155555000000000000000', '0110155555000100000000000'],
    10: ['5550050500505005050055500'],
    11: ['0020002020203020202000200'],
    12: ['6000022000555003333044444'],
    13: ['0005500055000000000500055', '0005500055000000600060000'],
    14: ['6666666666666666666666666'],
    15: ['0031300313003030031300303', '0000011313113030031300000', '0031001310013000031001300'],
    16: ['0000000600060600060000000', '0000006060000000606000000'] 
}

# Apply the correction to the DataFrames
df_human = correct_correct(df_human)
df_models = correct_correct(df_models)


### ERROR CODING
def categorize_error_type(x):
    x_lowercase = x.lower()
    if 'matrix' in x_lowercase:
        return 'matrix'
    elif 'concept' in x_lowercase:
        return 'concept'
    elif 'correct' in x_lowercase:
        return 'correct'
    elif 'duplicate' in x_lowercase or 'duplicated' in x_lowercase or 'copy' in x_lowercase:
        return 'part_duplicated'
    elif 'other' in x_lowercase:
        return 'other'
    elif 'wrong_item' in x_lowercase:
        return 'wrong_item'
    return np.nan 

def error_coding(df: pd.DataFrame, error_types_dict: dict) -> pd.DataFrame:
    '''
    Adds columns to the dataframe indicating the error type of the response.
    '''
    # Function to check error type
    def check_error_type(row, error_type):
        return int(error_types_dict[row['itemid']].get(row['response'], np.nan) == error_type)

    # Apply the check for each error type
    for error_type in ['matrix', 'concept', 'correct', 'part_duplicated', 'other', 'wrong_item']:
        df[f'error_{error_type}'] = df.apply(check_error_type, axis=1, error_type=error_type)
    
    # Check for empty responses
    df['error_empty'] = df['response'].apply(lambda x: int(x.count('0') == len(x)) if type(x) == str else 0)

    return df


# Create a dictionary of error types for each task based on the human rated error coding 
error_types_dict = {}
for task in range(1, 17):
    df = pd.read_excel('data/error_coding/responses_to_code.xlsx', sheet_name=f'Item {task}', converters={'response_id':str})
    df = df[~df['response_id'].isna()]
    df['Error Type'] = df['Error Type'].apply(categorize_error_type)
    error_types_dict[task] = df[['response_id', 'Error Type']].set_index('response_id').to_dict()['Error Type']

# Add error coding columns to the DataFrames
df_human = error_coding(df_human, error_types_dict)
df_models = error_coding(df_models, error_types_dict)


## Duplicated responses columns
def get_duplicated_responses(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Adds columns to the dataframe indicating whether the response is duplicated in 'A', 'B', 'C', 
    and another column for any duplicates.    
    '''
    # Function to check duplicates
    def check_duplicates(row, response_dict):
        return int(row['response'] in response_dict.get(row['itemid'], set()))

    # Apply the check for each column
    df['duplicate_A'] = df.apply(check_duplicates, axis=1, response_dict=items['A'].to_dict())
    df['duplicate_B'] = df.apply(check_duplicates, axis=1, response_dict=items['B'].to_dict())
    df['duplicate_C'] = df.apply(check_duplicates, axis=1, response_dict=items['C'].to_dict())

    # Combine the checks for any duplicate
    df['duplicated'] = df[['duplicate_A', 'duplicate_B', 'duplicate_C']].any(axis=1).astype(int)

    return df

df_human = get_duplicated_responses(df_human)

df_models['response'] = df_models.apply(lambda row: '0' * len(items.loc[row['itemid'], 'D']) if pd.isna(row['response']) else row['response'], axis=1) # Replace nan with zeros
df_models = get_duplicated_responses(df_models)


### EXCLUSION CRITERIA
def filter_responses(df, 
                     itemid_threshold=9, 
                     max_no_response=2):
    """
    Filters DataFrame based on itemid threshold and response criteria.
    """
    # Split DataFrame into SimpleARC and ConceptARC
    df_simple = df[df['itemid'] < itemid_threshold]
    df_concept = df[df['itemid'] >= itemid_threshold]

    # Function to filter out participants with too many no responses (more than 2)
    def filter_participants(df_sub, response_length, max_no_response):
        no_response = df_sub[df_sub['response'] == '0' * response_length]
        to_remove = no_response.groupby('respondent_id').filter(lambda x: len(x) > max_no_response)['respondent_id'].unique()
        return df_sub[~df_sub['respondent_id'].isin(to_remove)]

    # Apply filter to both kids and concept DataFrames
    df_simple_filtered = filter_participants(df_simple, response_length=9, max_no_response=max_no_response)
    df_concept_filtered = filter_participants(df_concept, response_length=25, max_no_response=max_no_response)

    return df_simple_filtered, df_concept_filtered


# Apply the exclusion filters to the DataFrames
df_human_simple, df_human_concept = filter_responses(df_human)
df_models_simple, df_models_concept = filter_responses(df_models)

# Print N
print(f'Models N: {df_models_simple['respondent_id'].nunique()} SimpleARC, {df_models_concept['respondent_id'].nunique()} ConceptARC')
print(f'Humans N: {df_human_simple['respondent_id'].nunique()} SimpleARC, {df_human_concept['respondent_id'].nunique()} ConceptARC')

### Save cleaned DataFrames
df_human_simple.to_csv('data/human/human_simple_clean.csv', index=False)
df_human_concept.to_csv('data/human/human_concept_clean.csv', index=False)
df_models_simple.to_csv('data/model/models_simple_clean.csv', index=False)
df_models_concept.to_csv('data/model/models_concept_clean.csv', index=False)


### Create Summary DataFrames

def summarize_df(df):
    agg_dict = {
        'correct': ('correct', 'sum'),
        'duplicated': ('duplicated', 'sum'),
        'matrix_error': ('error_matrix', 'sum'),
        'concept_error': ('error_concept', 'sum'),
        'part_duplicated_error': ('error_part_duplicated', 'sum'),
        'other_error': ('error_other', 'sum'),
        'empty_error': ('error_empty', 'sum'),
        'wrong_item_error': ('error_wrong_item', 'sum'),
    }
    
    # Age column for human dfs
    if 'age' in df.columns:
        agg_dict['age'] = ('age', 'first')

    return df.groupby('respondent_id').agg(**agg_dict).reset_index()

# Apply the function to each DataFrame
df_model_simple_summary = summarize_df(df_models_simple)
df_model_concept_summary = summarize_df(df_models_concept)
df_human_simple_summary = summarize_df(df_human_simple)
df_human_concept_summary = summarize_df(df_human_concept)


### Create bins for age theory
df_human_simple_summary['age_bin'] = pd.cut(df_human_simple_summary['age'], 
                                            bins=[3, 6, 9, 12, df_human_simple_summary['age'].max() + 1], 
                                            labels=['3-5', '6-8', '9-11', '12+'], 
                                            right=False, 
                                            include_lowest=True)
df_human_concept_summary['age_bin'] = pd.cut(df_human_concept_summary['age'], 
                                             bins=[6, 9, 12, df_human_concept_summary['age'].max() + 1],
                                             labels=['6-8', '9-11', '12+'], 
                                             right=False, 
                                             include_lowest=True)


n = df_human_simple_summary.groupby('age_bin')['age'].count().values
age_bin_labels = [f'Age {label} (N={n})' for label, n in zip(['3-5', '6-8', '9-11', '12+'], n)]


### Mean and standard deviation of correct responses across age and in LLMs (Table 1)
df_model_simple_summary['correct_scaled'] = df_model_simple_summary['correct']/8
round(df_model_simple_summary['correct'].mean()/8, 3), round(df_model_simple_summary['correct_scaled'].std(),3)
df_human_simple_summary['correct_scaled'] = df_human_simple_summary['correct']/8
round(df_human_simple_summary.groupby('age_bin')['correct'].mean()/8,3), round(df_human_simple_summary.groupby('age_bin')['correct_scaled'].std(),3)

df_model_concept_summary['correct_scaled'] = df_model_concept_summary['correct']/8
round(df_model_concept_summary['correct'].mean()/8, 3), round(df_model_concept_summary['correct_scaled'].std(),3)
df_human_concept_summary['correct_scaled'] = df_human_concept_summary['correct']/8
round(df_human_concept_summary.groupby('age_bin')['correct'].mean()/8,3), round(df_human_concept_summary.groupby('age_bin')['correct_scaled'].std(),3)



### PLOT Individual Strategies LLMs
def plot_summary(df_human_summary, model_summary_sorted, title, age_bin_labels):

    model_types = {
        'DiscoLM-mixtral-8x7b-v2': 'Fine-tuned',
        'Llama-2-7B-32K-Instruct': 'Base',
        'Mistral-7B-Instruct-v0.1': 'Base',
        'Mistral-7B-Instruct-v0.2': 'Base',
        'Mistral-7B-OpenOrca': 'Fine-tuned',
        'Mixtral-8x7B-Instruct-v0.1': 'Mixture of Experts',
        'MythoMax-L2-13b': 'Mixture of Experts',
        'Nous-Capybara-7B-V1p9': 'Fine-tuned',
        'Nous-Hermes-2-Yi-34B': 'Fine-tuned',
        'Nous-Hermes-Llama2-70b': 'Fine-tuned',
        'Nous-Hermes-llama-2-7b': 'Fine-tuned',
        'OpenHermes-2-Mistral-7B': 'Fine-tuned',
        'Platypus2-70B-instruct': 'Fine-tuned',
        'RedPajama-INCITE-7B-Chat': 'Base',
        'SOLAR-0-70b-16bit': 'Fine-tuned',
        'StripedHyena-Nous-7B': 'Base',
        'Toppy-M-7B': 'Mixture of Experts',
        'WizardLM-13B-V1.2': 'Fine-tuned',
        'Yi-34B-Chat': 'Base',
        'alpaca-7b': 'Fine-tuned',
        'openchat-3.5-1210': 'Fine-tuned',
        'vicuna-13b-v1.5': 'Fine-tuned',
        'vicuna-7b-v1.5': 'Fine-tuned',
        'gpt-4-vision': 'GPT',
        'gpt-3.5-turbo-0301': 'GPT',
        'gpt-4-0613': 'GPT',
    }

    model_types_colors = {
        'Base': 'slategrey',
        'Fine-tuned': 'midnightblue',
        'Mixture of Experts': 'darkgreen',
        'GPT': 'darkred',
        'Human': 'darkgoldenrod'
    }

    df_human_summary['copy'] = df_human_summary['duplicated'] + df_human_summary['part_duplicated_error']
    model_summary_sorted['copy'] = model_summary_sorted['duplicated'] + model_summary_sorted['part_duplicated_error']


    # human
    grouped = df_human_summary.groupby('age_bin').agg(['mean', 'sem'])

    human_avg_dup = grouped['copy']['mean'].values
    human_sem_dup = grouped['copy']['sem'].values

    human_avg_acc = grouped['correct']['mean'].values
    human_sem_acc = grouped['correct']['sem'].values

    human_avg_matrix_error = grouped['matrix_error']['mean'].values
    human_sem_matrix_error = grouped['matrix_error']['sem'].values

    human_avg_concept_error = grouped['concept_error']['mean'].values
    human_sem_concept_error = grouped['concept_error']['sem'].values

    # models
    model_acc = model_summary_sorted['correct'].values
    model_dup = model_summary_sorted['copy'].values
    model_matrix_error = model_summary_sorted['matrix_error'].values
    model_concept_error = model_summary_sorted['concept_error'].values
    model_names = model_summary_sorted['respondent_id'].to_list()


    #age bin n's
    n = df_human_summary.groupby('age_bin')['age'].count().values
    age_bin_labels = [f'Age {label} (N={n})' for label, n in zip(age_bin_labels, n)]

    dup_avg_all = np.concatenate([human_avg_dup, model_dup])
    acc_avg_all = np.concatenate([human_avg_acc, model_acc])
    matrix_error_avg_all = np.concatenate([human_avg_matrix_error, model_matrix_error])
    concept_error_avg_all = np.concatenate([human_avg_concept_error, model_concept_error])

    # Calculate the sum of the existing categories for each group
    sums = acc_avg_all + dup_avg_all + matrix_error_avg_all + concept_error_avg_all

    # Plot settings
    plt.figure(figsize=(20,9))
    bar_width = 0.35  # width of the bars
    index = np.arange(len(dup_avg_all))  # the label locations

    # Bars for mean number of duplicates
    plt.bar(index, acc_avg_all, bar_width, label='Correct Response', 
            color='firebrick', ecolor='black')

    # Bars for accuracy
    plt.bar(index, dup_avg_all, bar_width, label='Copy Error',
            bottom=acc_avg_all, 
            color='teal', ecolor='black')

    # Bars for matrix error
    plt.bar(index, matrix_error_avg_all, bar_width, label='Matrix Error',
            bottom=dup_avg_all + acc_avg_all,  
            color='goldenrod', ecolor='black')
    
    # Bars for concept error
    plt.bar(index, concept_error_avg_all, bar_width, label='Concept Error',
            bottom=dup_avg_all + acc_avg_all + matrix_error_avg_all,  
            color='rebeccapurple', ecolor='black')
    

    # Labels, title and custom x-axis tick labels, etc.
    plt.ylabel('Number of Responses', size=25)
    plt.title(title, size=25)
    plt.ylim(1, 8)  

    x_labels = age_bin_labels + model_names
    plt.xticks(index, x_labels, rotation=90, size=18)
    ax = plt.gca()  
    for i, label in enumerate(ax.get_xticklabels()):
        if i < len(age_bin_labels):
            label.set_color('goldenrod')
        else:
            label.set_color(model_types_colors[model_types[label.get_text()]])

    plt.yticks(size=22)
    bbox_to_anchor = (1.01, 1.3)  # Adjust as needed
    handles, labels = plt.gca().get_legend_handles_labels()

    new_order = [1, 0, 3, 2]  # This changes the order to flip columns
    handles = [handles[i] for i in new_order]
    labels = [labels[i] for i in new_order]

    # Create a 2x2 legend with flipped column order
    if title == 'KidsARC-Simple':
        plt.legend(handles, labels, frameon=False, fontsize=20, loc='upper right', bbox_to_anchor=bbox_to_anchor, ncol=2, columnspacing=1, handletextpad=0.5)
    else:
        plt.legend([], frameon=False)

    plt.gcf().subplots_adjust(bottom=0.45)
    plt.savefig(f'data/plots/{title}_ind_strategies.png', dpi=600)
    plt.show()

# Sort the models by duplicated (descending) and correct (ascending) in simpleARC
# Sort the models by the order of the sorted simpleARC models 
#df_model_simple_summary = df_model_simple_summary[df_model_simple_summary['respondent_id'].isin(df_model_concept_summary['respondent_id'])]
df_model_simple_summary = df_model_simple_summary.sort_values(by=['duplicated', 'correct'], ascending=[False, True])
order= [model for model in df_model_simple_summary['respondent_id'] if model in df_model_concept_summary['respondent_id'].to_list()]
df_model_concept_summary['respondent_id'] = pd.Categorical(df_model_concept_summary['respondent_id'], categories=order, ordered=True)
df_model_concept_summary = df_model_concept_summary.sort_values(by=['respondent_id'])

plot_summary(df_human_simple_summary, df_model_simple_summary[df_model_simple_summary['respondent_id'].isin(df_model_concept_summary['respondent_id'])], 'KidsARC-Simple', ['3-5', '6-8', '9-11', '12+'])
plot_summary(df_human_concept_summary, df_model_concept_summary, 'KidsARC-Concept', ['6-8', '9-11', '12+'])
trim_whitespace('data/plots/KidsARC-Simple_ind_strategies.png', 'data/plots/KidsARC-Simple_ind_strategies.png', margin=40)
trim_whitespace('data/plots/KidsARC-Concept_ind_strategies.png', 'data/plots/KidsARC-Concept_ind_strategies.png', margin=40)
join_images('data/plots/KidsARC-Simple_ind_strategies.png', 'data/plots/KidsARC-Concept_ind_strategies.png', 'data/plots/ARC_ind_strategies.png', join_direction='vertical')
trim_whitespace('data/plots/ARC_ind_strategies.png', 'data/plots/ARC_ind_strategies.png')


### PLOT Errors LLMs vs Humans
def plot_errors(df_human_summary, model_summary_sorted, title, age_bin_labels):

    respondent_colors = {
        '3-5': '#cc79a7',
        '6-8': '#d55e00',
        '9-11': '#009e73',
        '12+': '#0072b2',
        'model': '#d9cd2a'
    }

    df_human_summary['copy'] = df_human_summary['duplicated'] + df_human_summary['part_duplicated_error']
    model_summary_sorted['copy'] = model_summary_sorted['duplicated'] + model_summary_sorted['part_duplicated_error']

    # Error types
    error_types = ['Copy', 'Matrix', 'Concept', 'Empty', 'Other']
    human_avgs = [df_human_summary.groupby('age_bin')[error].mean().values for error in 
                  ['copy', 'matrix_error', 'concept_error', 'empty_error', 'other_error']]
    human_sems = [df_human_summary.groupby('age_bin')[error].sem().values for error in 
                  ['copy', 'matrix_error', 'concept_error', 'empty_error', 'other_error']]
    model_avgs = [model_summary_sorted[error].mean() for error in 
                  ['copy', 'matrix_error', 'concept_error', 'empty_error', 'other_error']]
    model_sems = [model_summary_sorted[error].sem() for error in 
                  ['copy', 'matrix_error', 'concept_error', 'empty_error', 'other_error']]
    
    # clip the error bars 
    # human_sems = [np.clip(human_avgs[i], 0.1, human_sems[i]) for i in range(len(human_avgs))]
    # model_sems = [np.clip(model_avgs[i], 0.3, model_sems[i]) for i in range(len(model_avgs))]

    age_n = df_human_summary.groupby('age_bin')['age'].count().values
    age_bin_labels_ = [f'Age {label} (N = {n})' for label, n in zip(age_bin_labels, age_n)]

    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 16})
    bar_width = 0.15  # Adjust bar width as needed
    n = len(error_types)
    index = np.arange(n) * (len(age_bin_labels) + 2.5) * bar_width  # Adjust the spacing between groups

    # Plot bars for each error type
    for i, error_type in enumerate(error_types):
        # Bar for model - always on the left
        plt.bar(index[i], model_avgs[i], bar_width, 
                yerr=model_sems[i], capsize=3, label=f'LLMs (N = {len(model_summary_sorted)})' if i == 0 else "_nolegend_",
                color=respondent_colors['model'], ecolor=respondent_colors['model'])

        # Bars for humans - to the right of the model bar
        for j in range(len(age_bin_labels)):
            plt.bar(index[i] + bar_width * (j + 1), human_avgs[i][j], bar_width, 
                    yerr=human_sems[i][j], capsize=3, label=age_bin_labels_[j] if i == 0 else "_nolegend_", 
                    color=respondent_colors[age_bin_labels[j]], ecolor=respondent_colors[age_bin_labels[j]])  # You can choose different colors

    # Labels, title, and custom x-axis tick labels
    sns.despine(bottom=True)
    plt.ylabel('Mean Number of Errors', size=25)
    plt.title(title, size=25)
    plt.ylim(0, 5) if title == 'KidsARC-Simple' else plt.ylim(0, 4)
    xtick_idx = index+0.3 if title == 'KidsARC-Simple' else index+0.25
    plt.xticks(xtick_idx, error_types, size=20)
    plt.tick_params(axis='x', which='both', length=0)
    plt.yticks(size=20)
    plt.legend(frameon=False, fontsize=20, loc='upper right')

    plt.savefig(f'data/plots/{title}_errors.png', dpi=600)

    # Show plot
    plt.show()


plot_errors(df_human_simple_summary, df_model_simple_summary, 'KidsARC-Simple', ['3-5', '6-8', '9-11', '12+'])
plot_errors(df_human_concept_summary, df_model_concept_summary, 'KidsARC-Concept', ['6-8', '9-11', '12+'])
join_images('data/plots/KidsARC-Simple_errors.png', 'data/plots/KidsARC-Concept_errors.png', 'data/plots/ARC_errors.png', join_direction='horizontal')
trim_whitespace('data/plots/ARC_errors.png', 'data/plots/ARC_errors.png')


### PLOT Item-by_Item responses
def plot_performance_responses(df_human: pd.DataFrame, df_models: pd.DataFrame, df_human_responses: pd.DataFrame, df_models_responses: pd.DataFrame, title: str, top=3, save=False) -> None:

    items, accuracy_human, sem_human = get_accuracy(df_human)
    _, accuracy_model, sem_model = get_accuracy(df_models)

    spacing_factor = 1.7

    # Reverse the order of items and their spacing for y-coordinates
    items_spaced = (items * spacing_factor)[::-1]

    plt.rcParams.update({'font.size': 16})
    sns.set_style('ticks')

    # Create the main plot
    fig, ax = plt.subplots(figsize=(10, 14))
    fig.subplots_adjust(left=0.3, bottom=0.15)

    # Draw bars in reversed order
    bars2 = ax.barh([i + 0.25 for i in items_spaced], accuracy_model, height=0.45, color='grey', label='LLMs')
    ax.errorbar(accuracy_model, [i + 0.25 for i in items_spaced], xerr=sem_model, fmt='none', ecolor='grey', capsize=2, linestyle='')
    bars = ax.barh([i - 0.25 for i in items_spaced], accuracy_human, height=0.45, color='teal', label='Humans')
    ax.errorbar(accuracy_human, [i - 0.25 for i in items_spaced], xerr=sem_human, fmt='none', ecolor='teal', capsize=2, linestyle='')


    ax.set_xlim(0, 1.3)
    ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))

    ax.tick_params(axis='y', which='both', left=False)
    # remove the y tick labels
    ax.set_yticklabels([])
    sns.despine(offset=10, trim=False, left=True)
    ax.set_title(title, color = 'black')
    ax.set_xlabel('Proportion Correct')
    ax.xaxis.set_label_coords(0.4, -0.05)


    # Add a secondary axes for annotations
    sec_ax = fig.add_axes([0.1, 0.2115, 0.1, 0.617])  
    sec_ax.set_ylim(items_spaced[0], items_spaced[-1])
    for spine in sec_ax.spines.values():
        spine.set_visible(False)
    sec_ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    sec_ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    sec_ax.yaxis.set_label_coords(-0.0, 0.5)

    # Add annotations to the main and secondary axes
    for idx, (item, bar) in enumerate(zip(items, bars)):

        # Annotations on the main plot
        incorrect = plot_incorrect_responses(df_human_responses, task=item, top=top)
        incorrect = trim_whitespace_numpy(incorrect)
        imagebox = OffsetImage(incorrect, zoom=0.11)  
        xy = (bar.get_width()+sem_human.iloc[idx]+0.15, (bar.get_y() + bar.get_height() / 2)+0.0)
        xybox_offset = (50, 0)
        ab = AnnotationBbox(imagebox, xy, xybox=xybox_offset, boxcoords="offset points", frameon=False)
        ax.add_artist(ab)

        # Annotations on the main plot
        incorrect = plot_incorrect_responses(df_models_responses, task=item, top=top)
        incorrect = trim_whitespace_numpy(incorrect)
        imagebox = OffsetImage(incorrect, zoom=0.11)  
        xy = (bars2[idx].get_width()+sem_model.iloc[idx]+0.15, (bar.get_y() + bar.get_height() / 2)+0.5)
        xybox_offset = (50, 0)
        ab = AnnotationBbox(imagebox, xy, xybox=xybox_offset, boxcoords="offset points", frameon=False, zorder=1)
        ax.add_artist(ab)

        # Annotations on the secondary axes
        task = plot_task(item, title=idx+1)
        imagebox = OffsetImage(task, zoom=0.1)
        xy = (0.8, items_spaced[::-1][idx])  # Center in the secondary axes
        ab = AnnotationBbox(imagebox, xy, frameon=False, xycoords='data', boxcoords=("axes fraction", "data"))
        sec_ax.add_artist(ab)

    #plt.hlines(y=1, xmin=1/1.5, xmax=1, color='black', linewidth=10, zorder=3)
    
    ax.legend(frameon=False, fontsize=20, bbox_to_anchor = (1.1, 1.1), loc='upper right')
    
    if save:
        fig.savefig(f'data/plots/{title}_Item_Performance.png', dpi=600)
    plt.show()
    plt.close()

plot_performance_responses(df_human_simple, df_models_simple, 
                           get_unique_responses(df_human_simple), get_unique_responses(df_models_simple), 'SimpleARC', save=True)

plot_performance_responses(df_human_concept, df_models_concept, 
                           get_unique_responses(df_human_concept), get_unique_responses(df_models_concept), 
                           'ConceptARC', save=True)
join_images('data/plots/SimpleARC_Item_Performance.png', 'data/plots/ConceptARC_Item_Performance.png', 'data/plots/ARC_Item_Performance.png', join_direction='horizontal')
trim_whitespace('data/plots/ARC_Item_Performance.png', 'data/plots/ARC_Item_Performance.png')


### Concept vs Matrix
concept_matrix_responses = {
    9: {'0110155555000000000000000': 'concept', '0110155555000100000000000':'matrix'},
    #15: {'0031300313003030031300303': 'concept', '0000011313113030031300000':'matrix', '0031001310013000031001300':'matrix'}
    15: {'0031300313003030031300303': 'concept', '0000011313113030031300000':'matrix'}
}

df_human_concept[(df_human_concept['itemid'] == 9) | (df_human_concept['itemid'] == 15)].apply(lambda row: concept_matrix_responses[row['itemid']].get(row['response'], np.nan), axis=1).dropna().value_counts().reset_index()
df_models_concept[(df_models_concept['itemid'] == 9) | (df_models_concept['itemid'] == 15)].apply(lambda row: concept_matrix_responses[row['itemid']].get(row['response'], np.nan), axis=1).dropna().value_counts().reset_index()


### DEMO
errors = ['000444000', '060444060', '000111000', '044444044']

error_names = ['Copy', 'Matrix', 'Concept', 'Other']

cmap, norm = color_pallette()

fig, axs = plt.subplots(1, 4, figsize=(2, 2))
axs = axs.flatten()
fig.subplots_adjust(left=0, right=1, top=1, bottom=0.01, wspace=0.4)

for i, (error, name) in enumerate(zip(errors, error_names)):
    matrix = response_to_array(error)
    axs[i].imshow(matrix, cmap=cmap, norm=norm)
    axs[i].grid(True, which='both', color='white', linewidth=0.5)
    axs[i].set_title(name, size=10)
    axs[i].set_yticks([x-0.5 for x in range(1 + len(matrix))])
    axs[i].set_xticks([x-0.5 for x in range(1 + len(matrix[0]))])
    axs[i].set_xticklabels([])
    axs[i].set_yticklabels([])
    axs[i].tick_params(axis='both', which='both', length=0)

#plt.tight_layout()
plt.savefig('data/plots/demo.png', dpi=600, bbox_inches='tight')
trim_whitespace('data/plots/demo.png', 'data/plots/demo.png')
