import sys
import numpy as np

LOCATION = 1
EVENT_TYPE_COUNT = 50
RES_TYPE_COUNT = 10
SEV_TYPE_COUNT = 5
LOG_FEATURE_COUNT = 400
FEATURE_COUNT = LOCATION + EVENT_TYPE_COUNT + RES_TYPE_COUNT + \
                SEV_TYPE_COUNT + LOG_FEATURE_COUNT

def extract_from_train(file_path):
    with open(file_path) as f:
        # skip header
        f.readline()
        for line in f:
            line = line.strip().split(',')
            id = int(line[0])
            loc = int(line[1].split(' ')[1])
            fault = int(line[2])
            yield id, loc, fault


def extract_from_test(file_path):
    with open(file_path) as f:
        # skip header
        f.readline()
        for line in f:
            line = line.strip().split(',')
            id = int(line[0])
            loc = int(line[1].split(' ')[1])
            yield id, loc

    
def extract_from_event(file_path):
    with open(file_path) as f:
        # skip header
        f.readline()
        for line in f:
            line = line.strip().split(',')
            id = int(line[0])
            event_type = int(line[1].split(' ')[1])
            yield id, event_type 


def extract_from_resource(file_path):
    with open(file_path) as f:
        # skip header
        f.readline()
        for line in f:
            line = line.strip().split(',')
            id = int(line[0])
            resource= int(line[1].split(' ')[1])
            yield id, resource


def extract_from_severity(file_path):
    with open(file_path) as f:
        # skip header
        f.readline()
        for line in f:
            line = line.strip().split(',')
            id = int(line[0])
            severity = int(line[1].split(' ')[1])
            yield id, severity


def extract_from_log(file_path):
    with open(file_path) as f:
        # skip header
        f.readline()
        for line in f:
            line = line.strip().split(',')
            id = int(line[0])
            log = int(line[1].split(' ')[1])
            logv = int(line[2])
            yield id, log, logv 



def construct(train, test, event, resource, severity, log):
    train_set = {}
    test_set = {}
    for id, loc, fault in extract_from_train(train):
        # feature count = id(1) + loc(1) + event(50) + 
        # res(10) + sev(5) + log(400) + class label(1)
        if id not in train_set:
            train_set.setdefault(id, [0 for i in range(FEATURE_COUNT+1)])
            train_set[id][0] = loc
            train_set[id][-1] = fault
        else:
            raise ValueError('id is not unique')

    for id, loc in extract_from_test(test):
        # feature count = id(1) + loc(1) + event(50) + 
        # res(10) + sev(5) + log(400) 
        if id not in test_set:
            test_set.setdefault(id, [0 for i in range(FEATURE_COUNT)])
            test_set[id][0] = loc
        else:
            raise ValueError('id is not unique')
    
    for id, event_type in extract_from_event(event):
        if id in train_set:
            train_set[id][event_type] = 1
        if id in test_set:
            test_set[id][event_type] = 1

    for id, res_type in extract_from_resource(resource):
        if id in train_set:
            train_set[id][EVENT_TYPE_COUNT+res_type] = 1
        if id in test_set:
            test_set[id][EVENT_TYPE_COUNT+res_type] = 1
            
        
    for id, sev_type in extract_from_severity(severity):
        if id in train_set:
            train_set[id][EVENT_TYPE_COUNT+RES_TYPE_COUNT+sev_type] = 1
        if id in test_set:
            test_set[id][EVENT_TYPE_COUNT+RES_TYPE_COUNT+sev_type] = 1

    for id, log, logv in extract_from_log(log):
        if id in train_set:
            train_set[id][EVENT_TYPE_COUNT+RES_TYPE_COUNT+SEV_TYPE_COUNT+log] = logv
        if id in test_set:
            test_set[id][EVENT_TYPE_COUNT+RES_TYPE_COUNT+SEV_TYPE_COUNT+log] = logv

    return train_set, test_set


def main():
    train = 'data/train.csv'
    test = 'data/test.csv'
    event = 'data/event_type.csv'
    resource = 'data/resource_type.csv'
    severity = 'data/severity_type.csv'
    log = 'data/log_feature.csv'

    train_set, test_set = construct(train, test, event, 
            resource, severity, log)

    print EVENT_TYPE_COUNT+RES_TYPE_COUNT+SEV_TYPE_COUNT+LOG_FEATURE_COUNT
    
    with open('train.dat', 'w') as f:
        for id in train_set:
            f.write('%s,%s\n' % (id, ','.join([str(item) for item in train_set[id]])))
    
    with open('test.dat', 'w') as f:
        for id in test_set:
            f.write('%s,%s\n' % (id, ','.join([str(item) for item in test_set[id]])))


if __name__ == '__main__':
    main()

