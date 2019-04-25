import os, sys
import json

def save_cfg_d(cfg_d={}, cfg_path_name='./ds_cfg_d.txt'):

    if os.path.exists(cfg_path_name):
        r = ''
        while not r in ['y', 'n']:
            r = input(' El archivo "{}" ya existe, desea sobreescribirlo? y/n: '.format(cfg_path_name))
            if not r in ['y', 'n']:
                print('Respuesta erronea "{}", intente nuevamente.'.format(r))
    else:
        r = 'y'
    
    if r == 'y':
        cfg_d_str = json.dumps(cfg_d)
        with open(cfg_path_name, 'w') as f:
            print(' Salvando:', cfg_path_name)
            f.write(cfg_d_str)

    return None

def load_cfg_d(cfg_path_name='./ds_cfg_d.txt'):
    with open(cfg_path_name, 'r') as f:
        print(' Restaurando:', cfg_path_name)
        cfg_d_str = ''.join(f.readlines())
    
    return json.loads(cfg_d_str)



if __name__ == '__main__':
    save_cfg_d({'a':6, 'b':88})

    d = load_cfg_d()
