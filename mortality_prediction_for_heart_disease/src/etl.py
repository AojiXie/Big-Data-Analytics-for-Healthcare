import utils
import pandas as pd
import numpy as np
from datetime import timedelta

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''

    events['group'] = np.where(events['patient_id'].isin(mortality['patient_id']), 'dead', 'alive')

    alive = events[events['group'] == 'alive']

    alive_df = alive[['patient_id', 'timestamp']].drop_duplicates()

    alive_df['timestamp'] = pd.to_datetime(alive_df['timestamp'])
    # alive_df['timestamp'] = alive_df['timestamp']

    dead_df = mortality[['patient_id', 'timestamp']]
    dead_df['timestamp'] = pd.to_datetime(dead_df['timestamp'])
    dead_df['timestamp'] = dead_df['timestamp'] - pd.to_timedelta(30, unit='d')

    alive_enddate = alive_df.groupby('patient_id').max()['timestamp'].reset_index()

    dead_enddate = dead_df

    indx_date = pd.concat([alive_enddate, dead_enddate])
    indx_date.columns = ['patient_id', 'indx_date']

    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', header=['patient_id', 'indx_date'], index=False)

    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''

    merged = pd.merge(indx_date, events, on='patient_id', how='inner')
    merged['timestamp'] = pd.to_datetime(merged['timestamp'])
    merged = merged[merged.timestamp <= merged.indx_date]
    merged = merged[merged.timestamp >= merged.indx_date - timedelta(days=2000)]
    filtered_events = merged[['patient_id', 'event_id', 'value']]
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', header=['patient_id', 'event_id', 'value'],
                           index=False)

    return filtered_events


def aggregate_events(filtered_events, mortality_df,feature_map, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''
    events_to_idx = pd.merge(filtered_events, feature_map, on='event_id')
    events_to_idx = events_to_idx[['patient_id', 'idx', 'value']]
    events_to_idx = events_to_idx.dropna()

    events_sum = events_to_idx[events_to_idx['idx'] < 2680]
    events_count = events_to_idx[events_to_idx['idx'] >= 2680]

    events_counts = events_count.groupby(['patient_id', 'idx']).agg('count')
    # events_counts.columns = ['patient_id', 'event_id', 'value']

    events_sums = events_sum.groupby(['patient_id', 'idx']).agg('sum')
    # events_sums.columns = ['patient_id', 'event_id', 'value']

    total_events = pd.concat([events_counts, events_sums])
    total_events.columns = ['value']
    total_events = total_events.reset_index()

    ##min- max
    total_events1 = total_events[['idx', 'value']]

    # min_events_value = total_events1.groupby(['idx']).min()

    max_events_value = total_events1.groupby(['idx']).max()

    # maxsubmin = max_events_value.sub(min_events_value)
    max_events_value = max_events_value.reset_index()
    max_events_value.columns = ['idx', 'max_value']
    # min_events_value = min_events_value.reset_index()
    # min_events_value.columns = ['idx', 'min_value']
    # maxsubmin = maxsubmin.reset_index()
    # maxsubmin.columns = ['idx', 'max-min']

    # normalized_df = pd.merge(min_events_value, maxsubmin, on='idx')
    # normalized_df = pd.merge(normalized_df, max_events_value, on='idx')

    df1 = pd.merge(total_events, max_events_value, on='idx')

    df1_not_zero = df1[df1['max_value'] != 0]
    df1_not_zero['value'] = df1_not_zero['value'] / df1_not_zero['max_value']

    df1_zero = df1[df1['max_value'] == 0]

    # df1_zero_events = df1_zero['idx'].value_counts()
    # df1_zero_events = df1_zero_events.reset_index()
    # df1_zero_events.columns = ['idx', 'counts']
    # df1_zero = pd.merge(df1_zero, df1_zero_events, on='idx')
    df1_zero['value'] = 1.0
    # df1_zero = df1_zero[['patient_id', 'idx', 'value', 'min_value', 'max-min']]

    aggregated_events = pd.concat([df1_zero, df1_not_zero])

    aggregated_events = aggregated_events[['patient_id', 'idx', 'value']]
    aggregated_events.columns = ['patient_id', 'feature_id', 'feature_value']

    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv',
                             header=['patient_id', 'feature_id', 'feature_value'],
                             index=False)

    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    aggregated_events['merged'] = aggregated_events.apply(lambda row: (row['feature_id'], row['feature_value']), axis=1)

    # aggregated_events['merged'] = aggregated_events.set_index('feature_id')['feature_value'].T.apply(tuple)
    # aggregated_events['merged'] = aggregated_events['feature_id'].astype(str) +':'+aggregated_events['feature_value'].astype(str)
    # aggregated_events['merged'] = aggregated_events['merged'].astype(float)
    patient_features = aggregated_events.groupby('patient_id')['merged'].apply(lambda x: x.tolist()).to_dict()

    events['group'] = np.where(events['patient_id'].isin(mortality['patient_id']), '1', '0')
    events = events.reset_index()
    events['group'] = events['group'].astype(int)

    mortality_df = events[['patient_id', 'group']]

    mortality = mortality_df.set_index('patient_id')['group'].to_dict()

    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')

    for key in sorted(patient_features):

        line1 = "%d" % (mortality[key])
        line2 = "%d %d" % (key, mortality[key])
        for value in sorted(patient_features[key]):
            merged = "%d:%.6f" % (value[0], value[1])
            line1 = line1 + " " + merged
            line2 = line2 + " " + merged
        deliverable1.write((line1 + " " + "\n").encode())
        deliverable2.write((line2 + " " + "\n").encode())

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()