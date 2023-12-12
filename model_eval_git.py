import numpy as np
import pandas as pd
import tensorflow as tf
import re
import os

# Sorry! This package is poorly optimized XD

# get_scaledPE_v1 : Ap1

# get_scaledPE_v2 : Ap2
# log_transform : Ap2
# inverse_log_transform : Ap2

# get_conditional_distribution_index_value : FNN for Ap1 and Ap2
# get_the_best_model_v1 : FNN Ap1
# get_the_best_model : FNN Ap2

# get_conditional_distribution_index_value_cgan_v1 : cGAN Ap1
# get_the_best_model_cgan_v1 : cGAN Ap1
# get_conditional_distribution_index_value_cgan : cGAN Ap2
# get_the_best_model_cgan : cGAN AP2


def log_transform(X, is_noise:float=None):
    if is_noise != None:
        return np.log(X+2)
    else:
        return np.log(X)

def inverse_log_transform(X_log_transformed, is_noise:float=None):
    if is_noise != None:
        return np.exp(X_log_transformed) - 2
    else:
        return np.exp(X_log_transformed)

def get_scaledPE_v1(target_generated:pd.DataFrame, target_ideal_mean:pd.DataFrame, target_ideal_var:pd.DataFrame):
    target_df_mean_comp = pd.concat([target_generated.groupby('QAM').mean(), target_ideal_mean], axis=1)
    target_df_per_diff_mean_real = (((target_df_mean_comp['real'] - target_df_mean_comp['ideal_real']) / target_df_mean_comp['ideal_real']) * 100).round(2)
    target_df_per_diff_mean_imaginary = (((target_df_mean_comp['imaginary'] - target_df_mean_comp['ideal_imaginary']) / target_df_mean_comp['ideal_imaginary']) * 100).round(2)
    target_df_per_diff_mean_dict = {'per_diff_real_%':target_df_per_diff_mean_real, 'per_diff_imaginary_%':target_df_per_diff_mean_imaginary}
    target_df_mean_comp_final = pd.concat([target_df_mean_comp, pd.DataFrame(target_df_per_diff_mean_dict)], axis=1)
    target_real_mean_avg = target_df_mean_comp_final['per_diff_real_%'].abs().mean()
    target_imaginary_mean_avg = target_df_mean_comp_final['per_diff_imaginary_%'].abs().mean()
    
    target_df_var_comp = pd.concat([target_generated.groupby('QAM').var(), target_ideal_var], axis=1)
    target_df_per_diff_var_real = (((target_df_var_comp['real'] - target_df_var_comp['ideal_real']) / target_df_var_comp['ideal_real']) * 100).round(2)
    target_df_per_diff_var_imaginary = (((target_df_var_comp['imaginary'] - target_df_var_comp['ideal_imaginary']) / target_df_var_comp['ideal_imaginary']) * 100).round(2)
    target_df_per_diff_var_dict = {'per_diff_real_%':target_df_per_diff_var_real, 'per_diff_imaginary_%':target_df_per_diff_var_imaginary}
    target_df_var_comp_final = pd.concat([target_df_var_comp, pd.DataFrame(target_df_per_diff_var_dict)], axis=1)
    target_real_var_avg = target_df_var_comp_final['per_diff_real_%'].abs().mean()
    target_imaginary_var_avg = target_df_var_comp_final['per_diff_imaginary_%'].abs().mean()

    return target_real_mean_avg, target_imaginary_mean_avg, target_real_var_avg, target_imaginary_var_avg

def get_scaledPE_v2(target_genuine_log_scaled:pd.DataFrame, target_generated_log_scaled:pd.DataFrame, is_noise:float,
                   target_ideal_mean:pd.DataFrame, target_ideal_var:pd.DataFrame):
    target_df_generated_original_scaled = target_generated_log_scaled.copy()
    target_df_genuine_original_scaled = target_genuine_log_scaled.copy()

    target_df_generated_original_scaled['data'] = inverse_log_transform(target_generated_log_scaled['data'], is_noise)
    target_df_genuine_original_scaled['data'] = inverse_log_transform(target_genuine_log_scaled['data'], is_noise)
    
    target_df_mean_comp = pd.concat([target_df_generated_original_scaled.groupby('d')['data'].mean(), target_ideal_mean], axis=1)
    target_df_per_diff_mean_real = (((target_df_mean_comp['data'] - target_df_mean_comp['ideal_data']) / target_df_mean_comp['ideal_data']) * 100).round(2)
    target_df_per_diff_mean_dict = {'mean_per_diff%':target_df_per_diff_mean_real, }
    target_df_mean_comp_final = pd.concat([target_df_mean_comp, pd.DataFrame(target_df_per_diff_mean_dict)], axis=1)
    target_mean_avg = target_df_mean_comp_final['mean_per_diff%'].abs().mean()
    
    target_df_var_comp = pd.concat([target_df_generated_original_scaled.groupby('d')['data'].var(), target_ideal_var], axis=1)
    target_df_per_diff_var_real = (((target_df_var_comp['data'] - target_df_var_comp['ideal_data']) / target_df_var_comp['ideal_data']) * 100).round(2)
    target_df_per_diff_var_dict = {'var_per_diff%':target_df_per_diff_var_real, }
    target_df_var_comp_final = pd.concat([target_df_var_comp, pd.DataFrame(target_df_per_diff_var_dict)], axis=1)
    target_var_avg = target_df_var_comp_final['var_per_diff%'].abs().mean()

    return target_mean_avg, target_var_avg


def get_conditional_distribution_index_value(target_FNN, condition, index):
    output = target_FNN.predict(condition)
    conditional_dist_index = tf.concat([output, index, condition[:, 1:]], axis=1)
    
    return conditional_dist_index

def get_the_best_model_v1(class_NakagamiV1, target_model_dir:str, custom_metric:dict, target_df_ideal_mean, target_df_ideal_var):

    if os.path.exists(f"{target_model_dir}/info_all_models.csv"):
        print("the csv file is alreay existed")
        return None
    
    list_master = list()
    
    list_file  = [epoch for epoch in os.listdir(target_model_dir)]
    list_numbers = list()
    for filename in list_file:
        match = re.search(r'(\d+)(?=\D*\.h5)', filename)
        if match:
            list_numbers.append(int(match.group(0)))
            
    epoch_max = max(list_numbers)
    list_epochs = list(range(1, epoch_max+1))
    # Generate file names with integers
    list_model_names = ['FNN-{0:04d}.h5'.format(i) for i in list_epochs]

    val_data_size = 100000*3
    
    val_nakagami_signal, val_nakagami_condition, val_nagakami_indices = class_NakagamiV1.generate(val_data_size)

    val_nagakami_indices = tf.cast(val_nagakami_indices, tf.float32)

    for target_epoch, target_model in zip(list_epochs, list_model_names):
        if custom_metric == None:
            target_model_loaded = tf.keras.models.load_model(f'{target_model_dir}/{target_model}')
        elif custom_metric is not None:
            target_model_loaded = tf.keras.models.load_model(f'{target_model_dir}/{target_model}', custom_objects=custom_metric)
            
        target_conditional_distribution_index_value = get_conditional_distribution_index_value(target_model_loaded, val_nakagami_condition, val_nagakami_indices)

        target_df_generated = pd.DataFrame(target_conditional_distribution_index_value[:, :3].numpy(), columns=['real', 'imaginary', 'QAM'])

        target_real_mean_avg, target_imaginary_mean_avg, target_real_var_avg, target_imaginary_var_avg = get_scaledPE_v1(
            target_df_generated, target_df_ideal_mean, target_df_ideal_var)
        
        list_master.append([target_epoch, target_real_mean_avg, target_imaginary_mean_avg, target_real_var_avg, target_imaginary_var_avg])

    target_df_result = pd.DataFrame(np.asarray(list_master), columns=['epoch', 'real_mean_avg', 'imaginary_mean_avg', 'real_var_avg', 'imaginary_var_avg'])
    
    target_df_result.to_csv(f"{target_model_dir}/info_all_models.csv", index=False)
    
    return target_df_result

def get_the_best_model(class_NakagamiV2, target_model_dir:str, custom_metric:dict, target_df_ideal_mean, target_df_ideal_var, is_noise:float=None):

    if os.path.exists(f"{target_model_dir}/info_all_models.csv"):
        print("the csv file is alreay existed")
        return None
    
    list_master = list()
    
    list_file  = [epoch for epoch in os.listdir(target_model_dir)]
    list_numbers = list()
    for filename in list_file:
        match = re.search(r'(\d+)(?=\D*\.h5)', filename)
        if match:
            list_numbers.append(int(match.group(0)))
            
    epoch_max = max(list_numbers)
    list_epochs = list(range(1, epoch_max+1))
    # Generate file names with integers
    list_model_names = ['FNN-{0:04d}.h5'.format(i) for i in list_epochs]
    
    val_data_size = 100000*3
    
    val_nakagami_signal, val_nakagami_condition, val_nakagami_indices = class_NakagamiV2.generate(val_data_size)

    for target_epoch, target_model in zip(list_epochs, list_model_names):
        if custom_metric == None:
            target_model_loaded = tf.keras.models.load_model(f'{target_model_dir}/{target_model}')
        elif custom_metric is not None:
            target_model_loaded = tf.keras.models.load_model(f'{target_model_dir}/{target_model}', custom_objects=custom_metric)
            
        target_conditional_distribution_index_value = get_conditional_distribution_index_value(target_model_loaded, val_nakagami_condition, val_nakagami_indices)

        target_df_genuine_log_scaled = pd.DataFrame(np.hstack((val_nakagami_signal, val_nakagami_indices)), columns=['data', 'd'])
        target_df_generated_log_scaled = pd.DataFrame(target_conditional_distribution_index_value.numpy(), columns=['data', 'd', 'r'])
        target_mean_avg, target_var_avg = get_scaledPE_v2(target_df_genuine_log_scaled, target_df_generated_log_scaled, 
                                                          is_noise, target_df_ideal_mean, target_df_ideal_var)

        list_master.append([target_epoch, target_mean_avg, target_var_avg])

    target_df_result = pd.DataFrame(np.asarray(list_master), columns=['epoch', 'mean_avg', 'var_avg'])
    
    target_df_result.to_csv(f"{target_model_dir}/info_all_models.csv", index=False)
    
    return target_df_result


def get_conditional_distribution_index_value_cgan_v1(target_generator, noise, condition, index):
    output = target_generator.predict([noise, condition])
    conditional_dist_index = tf.concat([output, index, condition[:, 2:]], axis=1)

    return conditional_dist_index
    
def get_the_best_model_cgan_v1(class_NakagamiV1_cgan, target_model_dir:str, save_interval:int, Z_dim:int, custom_metric:dict, target_df_ideal_mean, target_df_ideal_var):

    if os.path.exists(f"{target_model_dir}/info_all_models.csv"):
        print("the csv file is alreay existed")
        return None
    
    list_master = list()
    
    list_file  = [epoch for epoch in os.listdir(f"{target_model_dir}/gen")]
    # list_file
    list_numbers = list()
    for filename in list_file:
        match = re.search(r'(\d+)(?=\D*\.h5)', filename)
        if match:
            list_numbers.append(int(match.group(0)))
            
    epoch_max = max(list_numbers)
    list_epochs = list(range(0, epoch_max+1, save_interval))
    list_model_names = [f'generator_{i}.h5' for i in list_epochs]

    val_data_size = 100000*3
    
    val_nakagami_signal, val_nakagami_condition, val_nagakami_indices = class_NakagamiV1_cgan.generate(val_data_size)

    Z = np.random.normal(0, 1, size=(val_data_size, Z_dim))
    val_nagakami_indices = tf.cast(val_nagakami_indices, tf.float32)

    for target_epoch, target_model in zip(list_epochs, list_model_names):
        if custom_metric == None:
            target_model_loaded = tf.keras.models.load_model(f'{target_model_dir}/gen/{target_model}')
        elif custom_metric is not None:
            target_model_loaded = tf.keras.models.load_model(f'{target_model_dir}/gen/{target_model}', custom_objects=custom_metric)
            
        target_conditional_distribution_index_value = get_conditional_distribution_index_value_cgan_v1(target_model_loaded, Z, val_nakagami_condition, val_nagakami_indices)

        target_df_generated = pd.DataFrame(target_conditional_distribution_index_value[:, :3].numpy(), columns=['real', 'imaginary', 'QAM'])

        target_real_mean_avg, target_imaginary_mean_avg, target_real_var_avg, target_imaginary_var_avg = get_scaledPE_v1(
            target_df_generated, target_df_ideal_mean, target_df_ideal_var)
        
        list_master.append([target_epoch, target_real_mean_avg, target_imaginary_mean_avg, target_real_var_avg, target_imaginary_var_avg])

    target_df_result = pd.DataFrame(np.asarray(list_master), columns=['epoch', 'real_mean_avg', 'imaginary_mean_avg', 'real_var_avg', 'imaginary_var_avg'])
    
    target_df_result.to_csv(f"{target_model_dir}/info_all_models.csv", index=False)
    
    return target_df_result
    
def get_conditional_distribution_index_value_cgan(target_generator, noise, condition, index):
    output = target_generator.predict([noise, condition])
    conditional_dist_index = tf.concat([output, index, condition[:, 1:]], axis=1)
    
    return conditional_dist_index

def get_the_best_model_cgan(class_NakagamiV2_cgan, target_model_dir:str, save_interval:int, Z_dim:int, custom_metric:dict, target_df_ideal_mean, target_df_ideal_var, is_noise:float=None):

    if os.path.exists(f"{target_model_dir}/info_all_models.csv"):
        print("the csv file is alreay existed")
        return None
    
    list_master = list()
    
    list_file  = [epoch for epoch in os.listdir(f"{target_model_dir}/gen")]
    # list_file
    list_numbers = list()
    for filename in list_file:
        match = re.search(r'(\d+)(?=\D*\.h5)', filename)
        if match:
            list_numbers.append(int(match.group(0)))
            
    epoch_max = max(list_numbers)
    list_epochs = list(range(0, epoch_max+1, save_interval))
    list_model_names = [f'generator_{i}.h5' for i in list_epochs]

    val_data_size = 100000*3
    
    val_nakagami_signal, val_nakagami_condition, val_nakagami_indices = class_NakagamiV2_cgan.generate(val_data_size)

    Z = np.random.normal(0, 1, size=(val_data_size, Z_dim))
    val_nakagami_indices = tf.cast(val_nakagami_indices, tf.float32)

    for target_epoch, target_model in zip(list_epochs, list_model_names):
        if custom_metric == None:
            target_model_loaded = tf.keras.models.load_model(f'{target_model_dir}/gen/{target_model}')
        elif custom_metric is not None:
            target_model_loaded = tf.keras.models.load_model(f'{target_model_dir}/gen/{target_model}', custom_objects=custom_metric)
            
        target_conditional_distribution_index_value = get_conditional_distribution_index_value_cgan(target_model_loaded, Z, val_nakagami_condition, val_nakagami_indices)

        target_df_genuine_log_scaled = pd.DataFrame(np.hstack((val_nakagami_signal, val_nakagami_indices)), columns=['data', 'd'])
        target_df_generated_log_scaled = pd.DataFrame(target_conditional_distribution_index_value.numpy(), columns=['data', 'd', 'r'])
        target_mean_avg, target_var_avg = get_scaledPE_v2(target_df_genuine_log_scaled, target_df_generated_log_scaled, 
                                                          is_noise, target_df_ideal_mean, target_df_ideal_var)
        list_master.append([target_epoch, target_mean_avg, target_var_avg])

    target_df_result = pd.DataFrame(np.asarray(list_master), columns=['epoch', 'mean_avg', 'var_avg'])
    
    target_df_result.to_csv(f"{target_model_dir}/info_all_models.csv", index=False)
    
    return target_df_result





