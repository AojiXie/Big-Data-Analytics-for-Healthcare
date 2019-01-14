import time
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    TODO : This function needs to be completed.
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''
    events['group'] = np.where(events['patient_id'].isin(mortality['patient_id']), 'dead', 'alive')

    alive = events[events['group'] == 'alive']
    dead = events[events['group'] == 'dead']

    alive_counts = alive['patient_id'].value_counts()
    dead_counts = dead['patient_id'].value_counts()
    avg_dead_event_count = dead_counts.mean()
    max_dead_event_count = dead_counts.max()
    min_dead_event_count = dead_counts.min()
    avg_alive_event_count = alive_counts.mean()
    max_alive_event_count = alive_counts.max()
    min_alive_event_count = alive_counts.min()

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''
    events['group'] = np.where(events['patient_id'].isin(mortality['patient_id']), 'dead', 'alive')

    alive = events[events['group'] == 'alive']
    dead = events[events['group'] == 'dead']

    alive_encounter = alive[['patient_id', 'timestamp']].drop_duplicates()
    dead_encounter = dead[['patient_id', 'timestamp']].drop_duplicates()

    alive_encounter_count = alive_encounter['patient_id'].value_counts()
    dead_encounter_count = dead_encounter['patient_id'].value_counts()

    avg_dead_encounter_count = dead_encounter_count.mean()
    max_dead_encounter_count = dead_encounter_count.max()
    min_dead_encounter_count = dead_encounter_count.min()
    avg_alive_encounter_count = alive_encounter_count.mean()
    max_alive_encounter_count = alive_encounter_count.max()
    min_alive_encounter_count = alive_encounter_count.min()

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    events['group'] = np.where(events['patient_id'].isin(mortality['patient_id']), 'dead', 'alive')

    alive = events[events['group'] == 'alive']
    dead = events[events['group'] == 'dead']

    alive_df = alive[['patient_id', 'timestamp']].drop_duplicates()
    dead_df = dead[['patient_id', 'timestamp']].drop_duplicates()

    alive_df['timestamp'] = pd.to_datetime(alive_df['timestamp'])
    alive_df['timestamp'] = alive_df['timestamp'].dt.date
    dead_df['timestamp'] = pd.to_datetime(dead_df['timestamp'])
    dead_df['timestamp'] = dead_df['timestamp'].dt.date

    alive_startdate = alive_df.groupby('patient_id').min()['timestamp']
    alive_enddate = alive_df.groupby('patient_id').max()['timestamp']

    dead_startdate = dead_df.groupby('patient_id').min()['timestamp']
    dead_enddate = dead_df.groupby('patient_id').max()['timestamp']

    # dead_enddate.columns = ['patient_id', 'timestamp']

    alive_daterange = alive_enddate.sub(alive_startdate).dt.days

    dead_daterange = dead_enddate.sub(dead_startdate).dt.days

    avg_dead_rec_len = dead_daterange.mean()
    max_dead_rec_len = dead_daterange.max()
    min_dead_rec_len = dead_daterange.min()
    avg_alive_rec_len = alive_daterange.mean()
    max_alive_rec_len = alive_daterange.max()
    min_alive_rec_len = alive_daterange.min()

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following path variable in coding but switch it back when submission.
    train_path = '../data/train/'

    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute event count metrics: " + str(end_time - start_time) + "s"))
    print(event_count)

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute encounter count metrics: " + str(end_time - start_time) + "s"))
    print(encounter_count)

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute record length metrics: " + str(end_time - start_time) + "s"))
    print(record_length)
    
if __name__ == "__main__":
    main()
