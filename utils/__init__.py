import json

def load_json(file):
    try:
        with open(file, 'r') as f:
            return json.load(f)
    except:
        raise IOError('json file {} not found'.format(file))

def save_json(data, file):
    try:
        with open(file, 'w') as f:
            json.dump(data, f, indent=4)
    except:
        raise IOError('json file {} write error'.format(file))

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
