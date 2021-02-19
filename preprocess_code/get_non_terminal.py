# rewrite on 2018/1/8 by xxx, add parent

import json
import time
import joblib
import _pickle as pkl
from collections import Counter, defaultdict

from tqdm import tqdm

# attention line 42: for python dataset, not exclude the last one
train_filename = '../java_full/json_data/train_ast.json'
test_filename = '../java_full/json_data/test_ast.json'
valid_filename = '../java_full/json_data/valid_ast.json'
target_filename = '../java_full/pkl_data/non_terminal.pkl'

# global variables
typeDict = dict()  # map N's name into its original ID(before expanding into 4*base_ID)
numID = set()  # the set to include all sparse ID
no_empty_set = set()
typeList = list()  # the set to include all Types
numType = 0
dicID = dict()  # map sparse id to dense id (remove empty id inside 4*base_ID)


def process(filename):
    with open(filename, encoding='UTF-8') as lines:
        print('Start processing %s !!!' % filename)
        corpus_N = list()
        corpus_parent = list()

        for line in tqdm(lines):
            data = json.loads(line)
            line_N = list()
            has_sibling = Counter()
            parent_counter = defaultdict(lambda: 1)  # default parent is previous 1
            parent_list = list()

            if len(data) >= 3e4:
                continue

            for i, dic in enumerate(data):  # JS data[:-1] or PY data
                typeName = dic['type']
                if typeName in typeList:
                    base_ID = typeDict[typeName]
                else:
                    typeList.append(typeName)
                    global numType
                    typeDict[typeName] = numType
                    base_ID = numType
                    numType = numType + 1

                # expand the ID into the range of 4*base_ID, according to whether it has sibling or children.
                # Sibling information is got by the ancestor's children information
                if 'children' in dic.keys():
                    if has_sibling[i]:
                        ID = base_ID * 4 + 3
                    else:
                        ID = base_ID * 4 + 2

                    children = dic['children']
                    for j in children:
                        parent_counter[j] = j - i

                    if len(children) > 1:
                        for j in children:
                            has_sibling[j] = 1
                else:
                    if has_sibling[i]:
                        ID = base_ID * 4 + 1
                    else:
                        ID = base_ID * 4
                # recording the N which has non-empty T
                if 'value' in dic.keys():
                    no_empty_set.add(ID)

                line_N.append(ID)
                parent_list.append(parent_counter[i])
                numID.add(ID)

            corpus_N.append(line_N)
            corpus_parent.append(parent_list)
        return corpus_N, corpus_parent


def map_dense_id(data):
    result = list()
    for line_id in data:
        line_new_id = list()
        for i in line_id:
            if i in dicID.keys():
                line_new_id.append(dicID[i])
            else:
                dicID[i] = len(dicID)
                line_new_id.append(dicID[i])
        result.append(line_new_id)
    return result


def save(filename, vocab_size, trainData, validData, testData, trainParent, validParent,
         testParent):
    with open(filename, 'wb') as f:
        savings = {
            # 'typeDict': typeDict,
            # 'numType': numType,
            # 'dicID': dicID,
            'vocab_size': vocab_size,
            'trainData': trainData,
            'validData': validData,
            'testData': testData,
            'trainParent': trainParent,
            'validParent': validParent,
            'testParent': testParent,
            # 'typeOnlyHasEmptyValue': empty_set_dense,
        }
        pkl.dump(savings, f, protocol=2)


def main():
    start_time = time.time()
    trainData, trainParent = process(train_filename)
    validData, validParent = process(valid_filename)
    testData, testParent = process(test_filename)
    trainData = map_dense_id(trainData)
    testData = map_dense_id(testData)
    vocab_size = len(numID)
    assert len(dicID) == vocab_size

    # for print the N which can only has empty T
    assert no_empty_set.issubset(numID)
    empty_set = numID.difference(no_empty_set)
    empty_set_dense = set()
    # print(dicID)
    for i in empty_set:
        empty_set_dense.add(dicID[i])
    print('The N set that can only has empty terminals: ', len(empty_set_dense), empty_set_dense)
    print('The vocabulary:', vocab_size, numID)

    save(target_filename, vocab_size, trainData, validData, testData, trainParent,
         validParent, testParent)
    print('Finishing generating terminals and takes %.2fs' % (time.time() - start_time))


if __name__ == '__main__':
    main()
