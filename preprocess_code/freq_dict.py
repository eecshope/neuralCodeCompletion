# freq_dict: each terminal's frequency; terminal_num: a set about all the terminals.

import json
import time
import joblib
from collections import Counter
from tqdm import tqdm


# attention line 28: for python dataset, not exclude the last one
train_filename = '../java_full/train_ast.json'
target_filename = '../java_full/pkl_data/freq_dict.pkl'

freq_dict = Counter()
terminal_num = set()
terminal_num.add('EmptY')


def process(filename):
    with open(filename, encoding='UTF-8') as lines:
        print('Start processing %s !!!' % filename)
        for line in tqdm(lines):
            data = json.loads(line)
            if len(data) < 3e4:
                for i, dic in enumerate(data[:-1]):  # JS data[:-1] or PY data
                    if 'value' in dic.keys():
                        terminal_num.add(dic['value'])
                        freq_dict[dic['value']] += 1
                    else:
                        freq_dict['EmptY'] += 1


def save(filename):
    with open(filename, 'wb') as f:
        _save = {'freq_dict': freq_dict, 'terminal_num': terminal_num}
        joblib.dump(_save, f, protocol=2)


if __name__ == '__main__':
    start_time = time.time()
    process(train_filename)
    save(target_filename)
    print(freq_dict['EmptY'], freq_dict['Empty'], freq_dict['empty'], freq_dict['EMPTY'])
    print('Finishing generating freq_dict and takes %.2f' % (time.time() - start_time))
