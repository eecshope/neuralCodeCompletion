import json
import time
from tqdm import tqdm

train_filename = '../java_full/train_ast.json'
valid_filename = '../java_full/valid_ast.json'
test_filename = '../java_full/test_ast.json'


def process(filename):
    with open(filename, encoding='UTF-8') as lines:
        print('Start procesing %s !!!' % (filename))
        length = 0
        for line in tqdm(lines):
            data = json.loads(line)
            if len(data) < 3e4:
                length += len(data[:-1])  # total number of AST nodes
        return length


if __name__ == '__main__':
    start_time = time.time()
    train_len = process(train_filename)
    valid_len = process(valid_filename)
    test_len = process(test_filename)
    print('total_length is ', train_len + valid_len + test_len)
    print('Finishing counting the length and takes %.2f' % (time.time() - start_time))
