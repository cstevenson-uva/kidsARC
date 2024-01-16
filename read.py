import pandas as pd
import os

data_files = os.listdir('data/human')
data_files = [file for file in data_files if file.startswith('kidsarc_rows')]
data_files = sorted(data_files)

# concatanate all data files into one dataframe
data = pd.DataFrame()
for file in data_files:
    data = pd.concat([data, pd.read_csv('data/human/' + file)])

overall_n = len(data['participant_fk'].unique())

for sub in data['participant_fk'].unique():
    df = data[data['participant_fk'] == sub]
    print('participant_fk: ', sub)
    print('participant_fk: ', len(df))
    for item in df['itemid'].unique():
        df_1 = df[df['itemid'] == item]
        print('n_rows:', len(df_1))
        print('Age range:', df_1['age'].min(), df_1['age'].max())
        print('Unique rows:', len(df_1.iloc[:,2:].drop_duplicates()))
        print('')

df_nondup = pd.DataFrame()
for sub in data['participant_fk'].unique():
    df = data[data['participant_fk'] == sub]
    df_nondup = pd.concat([df_nondup, df.iloc[:,2:].drop_duplicates()])

df_nondup = df_nondup[df_nondup['saveitem_timestamp'] < '2023-12-11']
simple_arc, adult_arc = df_nondup[df_nondup['itemid'] < 9], df_nondup[df_nondup['itemid'] > 8]

df_nondup.groupby('itemid')['correct'].mean()




df = data[data['participant_fk'] == data['participant_fk'].unique()[3]]
df_2 = data[data['participant_fk'] == data['participant_fk'].unique()[2]]

list(df_2[df_2['itemid'] == 5]['age'].value_counts())

list(df[df['itemid'] == 5]['age'].value_counts()) == list(df_2[df_2['itemid'] == 5]['age'].value_counts())

df_2[df_2['itemid'] == 9]['age'].value_counts()


for i in data['kidsarc_response_id'].unique():
    df = data[data['kidsarc_response_id'] == i]
    print('kidsarc_response_id: ', i)
    print('n_rows:', len(df))
    print('Age range:', df['age'].min(), df['age'].max())
    print('Unique rows:', len(df.iloc[:,2:].drop_duplicates()))
    print('')
df['age'].value_counts()

     
# select the rows based on the index 
df = adult_arc[adult_arc['participant_fk'] == 4368]
index = df.iloc[:,2:].drop_duplicates().index
df = df.loc[index,:]

len(df.iloc[:,2:].drop_duplicates())

df = simple_arc[simple_arc['participant_fk'] == 4907]

# Assuming df is your DataFrame
original_row_count = len(df)
unique_row_count = len(df.drop_duplicates())

# Check if all rows are the same
all_rows_same = original_row_count == unique_row_count



df = simple_arc[simple_arc['kidsarc_response_id'] == 90]

len(df.iloc[:,2:].drop_duplicates())


df = pd.read_csv('data/human/' + data_files[0])


len(adult_arc['participant_fk'].unique())


