# According to the terminal_dict you choose (i.e. 5k, 10k, 50k), parse the json file and turn them into ids that are
# stored in pickle file Output just one vector for terminal, the upper part is the word id while the lower part is
# the location 0108 revise the Empty into EmptY, normal to NormaL Here attn_size matters

import json
import time
import h5py
import joblib

import numpy as np
from collections import deque


# attention line 48: for python dataset, not exclude the last one
terminal_dict_filename = '../java_full/pkl_data/terminal_dict.pkl'
train_filename = '../java_full/json_data/train_ast.json'
valid_filename = '../java_full/json_data/valid_ast.json'
test_filename = '../java_full/json_data/test_ast.json'
log_name = '../java_full/json_data/log.json'
data_name = '../java_full/java_full_for_pointer.h5'


def restore_terminal_dict(filename):
    with open(filename, 'rb') as f:
        savings = joblib.load(f)
        terminal_dict = savings['terminal_dict']
        terminal_num = savings['terminal_num']
        vocab_size = savings['vocab_size']
        return terminal_dict, terminal_num, vocab_size  # vocab_size is 50k, and also the unk_id


def process(filename, terminal_dict, unk_id, attn_size, verbose=False):
    with open(filename, encoding='UTF-8') as lines:
        print('Start processing %s !!!' % filename)
        terminal_length = list()
        terminal_corpus = list()
        attn_que = deque(maxlen=attn_size)
        attn_success_total = 0
        attn_fail_total = 0
        length_total = 0
        line_index = 0
        for line in lines:
            line_index += 1
            # if is_train and line_index == 11:
            #   continue
            if line_index % 1000 == 0:
                print('Processing line:', line_index)
            data = json.loads(line)
            if len(data) < 3e4:
                terminal_line = list()
                attn_que.clear()  # have a new queue for each file
                attn_success_cnt = 0
                attn_fail_cnt = 0
                for i, dic in enumerate(data):  # JS data[:-1] or PY data
                    if 'value' in dic.keys():
                        dic_value = dic['value']
                        if dic_value in terminal_dict.keys():  # take long time!!!
                            terminal_line.append(terminal_dict[dic_value])
                            attn_que.append('NormaL')
                        else:
                            if dic_value in attn_que:
                                location_index = \
                                    [len(attn_que) - ind for ind, x in enumerate(attn_que) if x == dic_value][-1]
                                location_id = unk_id + 1 + location_index
                                # print('\nattn_success!! its value is ', dic_value) print('The current file index:
                                # ', line_index, ', the location index', location_index,', the location_id: ',
                                # location_id, ',\n the attn_que', attn_que)
                                terminal_line.append(location_id)
                                attn_success_cnt += 1
                            else:
                                attn_fail_cnt += 1
                                terminal_line.append(unk_id)
                            attn_que.append(dic_value)
                    else:
                        terminal_line.append(terminal_dict['EmptY'])
                        attn_que.append('EmptY')
                terminal_corpus.append(terminal_line)
                terminal_length.append(len(terminal_line))
                attn_success_total += attn_success_cnt
                attn_fail_total += attn_fail_cnt
                attn_total = attn_success_total + attn_fail_total
                length_total += len(data)
                # print ('Process line', line_index, 'attn_success_cnt', attn_success_cnt, 'attn_fail_cnt',
                # attn_fail_cnt,'data length', len(data))
                if verbose and line_index % 1000 == 0:
                    print(
                        '\nUntil line %d: attn_success_total: %d, attn_fail_total: %d, success/attn_total: %.4f, '
                        'length_total: %d, attn_success percentage: %.4f, total unk percentage: %.4f\n' %
                        (line_index, attn_success_total, attn_fail_total, float(attn_success_total) / attn_total,
                         length_total,
                         float(attn_success_total) / length_total, float(attn_total) / length_total))
        with open('output.txt', 'a') as fout:
            fout.write(
                'Statistics: attn_success_total: %d, attn_fail_total: %d, success/fail: %.4f, length_total: %d, '
                'attn_success percentage: %.4f, total unk percentage: %.4f\n' %
                (attn_success_total, attn_fail_total, float(attn_success_total) / attn_fail_total, length_total,
                 float(attn_success_total) / length_total, float(attn_success_total + attn_fail_total) / length_total))

        return terminal_corpus, terminal_length


def save(db, data, length, name):
    data = np.concatenate([np.asarray(ele) for ele in data])
    db.create_dataset(name+"Data", data=data)
    db.create_dataset(name+"Length", data=np.asarray(length))


def main():
    start_time = time.time()
    attn_size = 50
    terminal_dict, terminal_num, vocab_size = restore_terminal_dict(terminal_dict_filename)
    trainData = process(train_filename, terminal_dict, vocab_size, attn_size=attn_size, verbose=True)
    validData = process(valid_filename, terminal_dict, vocab_size, attn_size=attn_size, verbose=True)
    testData = process(test_filename, terminal_dict, vocab_size, attn_size=attn_size, verbose=True)

    with h5py.File(data_name, "w") as db:
        save(db, trainData[0], trainData[1], "train")
        save(db, validData[0], validData[1], "valid")
        save(db, testData[0], testData[1], "test")

    print('Finishing generating terminals and takes %.2f' % (time.time() - start_time))


if __name__ == '__main__':
    main()
