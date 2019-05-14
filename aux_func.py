import os, sys
import json


def make_dir_path(path='./algo1/algo2', verbose=True):
    path_v = path.replace('\\','/').split('/')

    for i in range(len(path_v)):
        p = '/'.join(path_v[:i+1])
        if len(p) > 0 and not os.path.exists(p):
            if verbose:
                print(' - make_dir_path: Creando:', p)
            os.mkdir(p)
    return None
    
    
def show_diff(cfg_d, old_cfg_d, i_level=0):
    
    keys_v = sorted(list( set(cfg_d.keys()).union( set(old_cfg_d.keys()) ) ))

    n_changes = 0
    for k in keys_v:
        if k in cfg_d.keys() and k in old_cfg_d.keys():
            if cfg_d[k] != old_cfg_d[k]:
                if type(cfg_d[k]) is dict and type(old_cfg_d[k]) is dict:
                    print('{} |-> {:10s}'.format(i_level*'    ', k))
                    n_changes += show_diff(cfg_d[k], old_cfg_d[k], i_level+1)
                else:
                    print('{} |-> {:10s}: \t  {:15s} >>> {:15s} '.format(i_level*'    ', k, str(old_cfg_d[k]), str(cfg_d[k]) ))
                    n_changes += 1

        if k not in cfg_d.keys():
            print('{} |-> {:10s}: \t  {:15s} >>> {:15s} '.format(i_level*'    ', k, str(old_cfg_d[k]), 'ERASED!!'))
            n_changes += 1

        if k not in old_cfg_d.keys():
            print('{} |-> {:10s}: \t  {:15s} >>> {:15s} '.format(i_level*'    ', k, 'EMPTY!!', str(cfg_d[k])))
            n_changes += 1

    return n_changes
    
def load_cfg_d(cfg_path_name='./ds_cfg_d.txt'):
    cfg_path_name = cfg_path_name.replace('\\','/')
    
    with open(cfg_path_name, 'r') as f:
        print(' Restaurando:', cfg_path_name)
        cfg_d_str = ''.join(f.readlines())
    
    return json.loads(cfg_d_str)


def save_cfg_d(cfg_d={}, cfg_path_name='./ds_cfg_d.txt'):
    cfg_path_name = cfg_path_name.replace('\\','/')
    
    path_dir, filename = os.path.split(cfg_path_name)
    make_dir_path(path_dir)
    
    if os.path.exists(cfg_path_name):
        old_cfg_d = load_cfg_d(cfg_path_name)
        cfg_d     = json.loads(json.dumps(cfg_d))

        if old_cfg_d != cfg_d:
            r = ''
            while not r in ['y', 'n']:
                print(' El archivo "{}" ya existe, y a cambiado:'.format(cfg_path_name))
                show_diff(cfg_d, old_cfg_d)
                print(' Desea actualizar la configuracion?? (y/n) ', end='')
                r = input()
                
                if not r in ['y', 'n']:
                    print('Respuesta erronea "{}", intente nuevamente.'.format(r))
        else:
            r = 'n'
    else:
        r = 'y'
    
    if r == 'y':
        cfg_d_str = json.dumps(cfg_d)
        with open(cfg_path_name, 'w') as f:
            print(' Salvando:', cfg_path_name)
            f.write(cfg_d_str)

    return None



if __name__ == '__main__':
    save_cfg_d({'a':88, 'b':6, 'c': 5, 'd':{'r':2, 'm':3}})

    d = load_cfg_d()
