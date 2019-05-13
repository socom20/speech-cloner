#! /usr/bin/python3
import os, sys


def remove_group(i, files_v, folder_name, ext_v):
    for e in ext_v:
        file_path = os.path.join(folder_name, files_v[i][0] + e)
        try:
            os.remove(file_path)
        except Exception as e:
            print(' - ERROR, while deleting "{}": {}'.format(file_path, e), file=sys.stderr)
            r = ''
            while r not in ('y', 'n'):
                print(' Continue (y/n):', end='')
                r = input()

            if r == 'y':
                continue
            else:
                sys.exit(1)
                
##        print(file_path)
        
    return None
    

if __name__ == '__main__':
    folder_name = './dec_ckpt'
    ext_v = ['.data-00000-of-00001', '.index', '.meta']
    step_min = 10000
    n_saves  = 100


    print(' Cleaning:', folder_name)
    print(' Keep at least:', n_saves, 'checkpoints.')

    print('ENTER to continue ... ', end='')
    input()
    n_deleted = 0
    files_v = [(f.split(ext_v[0])[0], int(f.split(ext_v[0])[0].split('-')[1])) for f in os.listdir(folder_name) if ext_v[0] in f]
    files_v.sort(key=lambda l: l[1])

    print('Total Files:', len(files_v))

    i = 0
    while i < len(files_v):
        if files_v[i][1] < step_min:
            remove_group(i, files_v, folder_name, ext_v)
            n_deleted += 1
            del(files_v[i])
        else:
            i += 1


    n_rest = len(files_v)
    delta  = max(n_rest//n_saves, 1)


    idx_left_v = [0]
    while idx_left_v[-1]+delta < n_rest:
        idx_left_v.append(idx_left_v[-1]+delta)

    if idx_left_v[-1] != n_rest-1:
        idx_left_v.append(n_rest-1)

    for i in range(len(idx_left_v)-1):
        for i_d in range(idx_left_v[i]+1, idx_left_v[i+1]):
            remove_group(i_d, files_v, folder_name, ext_v)
            n_deleted += 1
            
    print('Deleted files:', n_deleted)
    print('Files left =', len(idx_left_v))
    
    print('ENTER to finish ... ', end='')
    input()

        
