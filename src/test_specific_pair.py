
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.layers import serialize
from attacks.universal_perturbation import Universal_perturbation
from attacks.CW import CarliniWagnerL2
from data.cifar10 import CifarData
# from data.GTSRB import GTSRBData
from defences.activation_clustering import ActivationClustering
from defences.spectral import compute_corr
from model.cifar10 import CifarModel
# from model.GTSRBModel import GTSRBModel
from visualization import *
import numpy as np
import glob
import argparse


def test_deserialize_model(param, args):
    # the json configuration can be accessed by model.param
    K.set_learning_phase(1)
    if param.get_conf('model_prefix') == 'cifar':
        Data = CifarData
        Model = CifarModel
    elif param.get_conf('model_prefix') == 'GTSRB':
        Data = GTSRBData
        Model = GTSRBModel
    
    path = param.get_conf('model_path_finetune')
    if os.path.exists(path) and 'pkl' in path:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        model.set_param(param)
        print('model load success')
    else:
        model = Model(param)
        data_clean = Data(param)
        data_clean.load_data(is_add_channel=True)
        model.init(data_clean)
        model.init_model()
        model.train(data_clean, nb_epochs=10)
        serialize_name = '_'.join(
        [param.get_conf('model_prefix'), 'clean', get_date()]) + '.pkl'
        print('serialize_name = ', serialize_name)
        with open(serialize_name,'wb') as f:
            pickle.dump(model,f)
        param.set_conf('model_path_finetune', serialize_name)
    
    data = Data(param)
    data.load_data()
    data.gen_backdoor(model)

    model.train(data)
    
    
    serialize_name = '_'.join(
        [param.get_conf('model_prefix'), get_date()]) + \
         '_poison_{}to{}'.format(param.get_conf('poison_label_source'),param.get_conf('poison_label_target')) + '.pkl'
    print('serialize_name = ', serialize_name)
    serialize_path = os.path.join(param.get_conf('save_dir'), serialize_name)
    with open(serialize_path,'wb') as f:
        pickle.dump(model,f)

    K.set_learning_phase(0)
    with open(serialize_path, 'rb') as f:
        model2 = pickle.load(f)

    model2.predict(data)
    ac = ActivationClustering(data, param, model2)
    ac.relative_size_metric()



def test_resume_model(param, args):
    # the json configuration can be accessed by model.param    
    K.set_learning_phase(0)
    with open(param.get_conf('model_path_backdoor'), 'rb') as f:
        model = pickle.load(f)
    model.param.set_conf('pert_path', param.get_conf('pert_path'))
    print("model", param.get_conf('model_path_backdoor'), 'load success')

    if model.param.get_conf('model_prefix') == 'GTSRB':
        data = GTSRBData(model.param)
    elif model.param.get_conf('model_prefix') == 'cifar':
        data = CifarData(model.param)
    
    data.load_data()
    data.restore_backdoor(model)
    # data.gen_backdoor()
    model.param.print_conf()
    print('model load success')

    model.predict(data)
    if args.spectral:
        compute_corr(model.param, model, data)
    else:
        ac = ActivationClustering(data, model.param, model)
        # ac.size_metric()
        ac.relative_size_metric()

def gen_perturbation(model, data, method, param):

    # serialize perturbation

    if method == 'universal':
        universal = Universal_perturbation(model, param)
        v = universal.universal_perturbation(data, param.get_conf('poison_label_source'),
                                             param.get_conf('poison_label_target'), xi= param.get_conf('pert_xi')/255.0)
        serialize_name = 'universal_{}to{}'.format(param.get_conf('poison_label_source'),
                                        param.get_conf('poison_label_target'))
        serialize_name = universal.serialize(serialize_name)
    elif method == 'CW':
        cw = CarliniWagnerL2(model, param)
        cw.attack(data,xi= param.get_conf('pert_xi') / 255.0)
        serialize_name = 'CW_{}to{}'.format(param.get_conf('poison_label_source'),
                                param.get_conf('poison_label_target'))
        serialize_name = cw.serialize(serialize_name)
    # 3. test perturbation on some input cases
    return serialize_name




def gen_rand_perturbation(param, args):
    model_path = param.get_conf('model_path')
    K.set_learning_phase(0)
    if param.get_conf('model_prefix') == 'GTSRB':
        data = GTSRBData(param)
        model = GTSRBModel
    else:
        data = CifarData(param)
        model = CifarModel
    data.load_data(is_add_channel=True)
    if os.path.exists(model_path) and 'pkl' in model_path:
        # 2. load model and generate perturbation
        serialize_path = model_path
        with open(serialize_path, 'rb') as f:
            model = pickle.load(f)
    else:
                
        model = model(param)
        model.init(data)
        model.init_model()
        model.train(data, nb_epochs=10)
        serialize_name = '_'.join(
            [param.get_conf('model_prefix'), get_date()]) + '_clean.pkl'
        print('serialize_name = ', serialize_name)
        serialize_path = os.path.join(
            param.get_conf('save_dir'), serialize_name)

        with open(serialize_path, 'wb') as f:
            pickle.dump(model, f)
        print('model dump success')
    # model.predict(data)
    print('model load success')
   
    param.set_conf('poison_label_source', args.source)
    param.set_conf('poison_label_target', args.target)
    print('source number is {}, target number is {}'.format(args.source, args.target))
    method = param.get_conf('method')

    return gen_perturbation(model, data, method=method, param=param, )
    

def experiment_on_pair(args, param, pair):

    param.set_conf('poison_label_source', pair[0])
    param.set_conf('poison_label_target', pair[1])
    test_deserialize_model(param, args)

    K.clear_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='random pair testing')
    parser.add_argument('-c', '--config', default='train.json', type=str, help='config file')
    parser.add_argument('-g', '--gen', action='store_true', help='generate perturbations')
    parser.add_argument('-s','--source', type=int, default=5, help='attack spurce')
    parser.add_argument('-t','--target', type=int, default=6, help='attack target')
    parser.add_argument('-r', '--resume', action='store_true', help="resume and test trained models")
    parser.add_argument('-sp', '--spectral', action='store_true', help="using spectral to detect")
    args = parser.parse_args()
    json_name = args.config
    param = Param(json_name)
    param.load_json()
    if args.gen:
        pert_path = gen_rand_perturbation(param, args)
        param.set_conf("pert_path", pert_path)
    if args.resume:
        test_resume_model(param, args)
    else:
        experiment_on_pair(args, param, (args.source, args.target))


            
