from __future__ import print_function
import numpy as np
import os, sys
import subprocess
import time
import shutil
import json
import argparse

def gen_command(script, d):
    command = ['python3', script]
    for i in d:
        name = '--' + i + ' '
        value = d[i]
        opt = name + str(value)
        command.append(opt)
    command = ' '.join(command)

    return command

def save_code(output_code_path):
    curr_src_path = './'
    curr_config_path = '../config/'
    output_src_path = output_code_path + '/src'
    output_config_path = output_code_path + '/config'
    line1 = 'cp ' + curr_src_path + '* ' + output_src_path
    line2 = 'cp ' + curr_config_path + '* ' + output_config_path
    copy1 = subprocess.Popen(line1, shell=True)
    copy1.communicate()
    copy2 = subprocess.Popen(line2, shell=True)
    copy2.communicate()


def run_experiment(num_experiment=0, num_run=0, num_folds=2,
                   dataset='iemocap_randsplit', experiment_folder='../temp/',
                   script='training_autoencoder.py', parameters={}):
    '''
    run the crossvalidation
    '''
    print("NEW EXPERIMENT: exp: " + str(num_experiment) + ' run: ' + str(num_run))
    print('Dataset: ' + dataset)

    #create output path if not existing
    output_path = experiment_folder + '/experiment_' + str(num_experiment)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_temp_path = output_path + '/temp'
    if not os.path.exists(output_temp_path):
        os.makedirs(output_temp_path)

    output_models_path = output_path + '/models'
    if not os.path.exists(output_models_path):
        os.makedirs(output_models_path)

    output_results_path = output_path + '/results'
    if not os.path.exists(output_results_path):
        os.makedirs(output_results_path)


    output_temp_data_path = output_temp_path + '/temp_data'
    if not os.path.exists(output_temp_data_path):
        os.makedirs(output_temp_data_path)

    output_temp_results_path = output_temp_path + '/temp_results'
    if not os.path.exists(output_temp_results_path):
        os.makedirs(output_temp_results_path)

    output_code_path = output_path + '/code'
    if not os.path.exists(output_code_path):
        os.makedirs(output_code_path)

    output_src_path = output_code_path + '/src'
    if not os.path.exists(output_src_path):
        os.makedirs(output_src_path)

    output_config_path = output_code_path + '/config'
    if not os.path.exists(output_config_path):
        os.makedirs(output_config_path)


    #initialize results dict
    folds = {}

    #iterate folds
    for i in range(num_folds):
        #unroll parameters to find task_type:
        #create paths
        num_fold = i

        #init paths
        model_name = output_models_path + '/model_xval_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) + '_fold' + str(num_fold)
        results_name = output_temp_results_path + '/temp_results_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) + '_fold' + str(num_fold) + '.npy'
        #parameters_name = output_parameters_path + '/parameters_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) +  '.txt'

        #init results as ERROR
        np.save(results_name, np.array(['ERROR']))

        #run training
        time_start = time.perf_counter()

        parameters['num_folds'] = num_folds
        parameters['num_fold'] = num_fold
        parameters['model_path'] = model_name
        parameters['results_path'] = results_name

        shell_command = gen_command(script, parameters)

        print('\nShell command:\n')
        print (shell_command)
        training = subprocess.Popen(shell_command, shell=True)


        training.communicate()
        training.wait()


        training_time = (time.perf_counter() - time_start)
        print ('training time: ' + str(training_time))

        #wait for file to be created
        flag = 'ERROR'
        while flag == 'ERROR':
            time.sleep(0.2)
            flag = np.load(results_name, allow_pickle=True)

        #update results dict
        temp_results = np.load(results_name, allow_pickle=True)
        temp_results = temp_results.item()
        folds[i] = temp_results
        #END OF FOLD ITERATION

    #compute summary
    #compute mean loss and loss std of values across folds
    keys = list(folds[0].keys())
    keys = [i for i in keys if 'parameters' not in i and 'hist' not in i]  #remove non-values entries
    upper_keys = list(folds.keys())
    folds['summary'] = {}

    for f in upper_keys:
        for k in keys:
            if not k in folds['summary']:
                folds['summary'][k] = []
                folds['summary'][k].append(folds[f][k])
            else:
                folds['summary'][k].append(folds[f][k])
    for k in list(folds['summary'].keys()):
        mean_k = k + '_mean'
        std_k = k + '_std'
        folds['summary'][mean_k] = np.mean(folds['summary'][k])
        folds['summary'][std_k] = np.std(folds['summary'][k])

    print (folds['summary'])


    #save results dict
    dict_name = 'results_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) + '.npy'
    final_dict_path = output_results_path + '/' + dict_name
    np.save(final_dict_path, folds)

    #generate results spreadsheet
    spreadsheet_name = dataset + '_exp' + str(num_experiment) + '_results_spreadsheet.xls'
    output_file = os.path.join(output_results_path,spreadsheet_name)
    spreadsheet_parameters = {'input_folder':output_results_path,
                              'output_name':output_file,
                              'profile': parameters['spreadsheet_profile']}
    gen_xls_string = gen_command(script='results_to_excel.py', d=spreadsheet_parameters)
    gen_spreadsheet = subprocess.Popen(gen_xls_string, shell=True)
    gen_spreadsheet.communicate()
    gen_spreadsheet.wait()

    #save current code
    save_code(output_code_path)

'''
def run_experiment(num_experiment=0, num_run=0, num_folds=2,
                   dataset='iemocap_randsplit', experiment_folder='../temp/',
                   script='training_autoencoder.py', parameters={}):
'''
def grid_search(experiments_folder, output_folder, ids, begin, end, gpu_id):
    contents = os.listdir(experiments_folder)
    selected_experiments = [i for i in contents if int(i.split('_')[0]) in ids]
    print (selected_experiments)
    #iterate experiments
    for exp in selected_experiments:
        #read json
        file = os.path.join(experiments_folder, exp)
        with open(file) as json_file:
            parameters = json.load(json_file)

        #exp_keys = [i for i in parameters if 'global_parameters' not in i]
        #append global parameters in every experiment instance
        exp_keys = list(range(begin,end+1))
        exp_keys = [str(i) for i in exp_keys]
        print (exp_keys)

        for run in exp_keys:
            for g in parameters['global_parameters']:
                name = g
                value = parameters['global_parameters'][g]
                parameters[run][name] = value
            parameters[run]['gpu_id'] = gpu_id
            #run experiment instance with correct parameters

            run_experiment(num_experiment=exp,
                           num_run=run,
                           num_folds=parameters[str(run)]['num_folds'],
                           dataset=parameters[str(run)]['dataset'],
                           experiment_folder=output_folder,
                           script=parameters[str(run)]['script'],
                           parameters=parameters[str(run)]
                           )

    print ('\nALL EXPERIMENTS REQUESTED COMPLETED')




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments_folder', type=str, default='experiments_quat')
    parser.add_argument('--output_folder', type=str, default='../new_experiments')
    parser.add_argument('--ids', type=str, default='[608]')
    parser.add_argument('--first', type=int, default=1)
    parser.add_argument('--last', type=int, default=12)
    parser.add_argument('--gpu_id', type=int, default=1)
    args = parser.parse_args()


    grid_search(experiments_folder=args.experiments_folder,
                output_folder=args.output_folder,
                ids=eval(args.ids),
                begin=args.first,
                end=args.last,
                gpu_id=args.gpu_id)
