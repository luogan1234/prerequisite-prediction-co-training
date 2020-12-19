import numpy as np
import os
import json

def stat():
    files = os.listdir('result/result/')
    files.sort()
    for file in files:
        ps, rs, f1s = [], [], []
        with open('result/result/'+file, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                p, r, f1 = float(obj['p']), float(obj['r']), float(obj['f1'])
                if p < 0.5 or r < 0.5 or f1 < 0.5:
                    continue
                ps.append(p)
                rs.append(r)
                f1s.append(f1)
        print(file)
        print('average p: {:.3f}, average r: {:.3f}, average f1: {:.3f}\n'.format(np.mean(ps), np.mean(rs), np.mean(f1s)))

if __name__ == '__main__':
    stat()