""" Imports """

import pickle

""" Helper functions """

def save_tensor(koopman_tensor, env_id, saved_model_name):
    with open(f'./koopman_tensor/saved_models/{env_id}/{saved_model_name}.pickle', 'wb') as handle:
        pickle.dump(koopman_tensor, handle)

def load_tensor(env_id, saved_model_name):
    with open(f'./koopman_tensor/saved_models/{env_id}/{saved_model_name}.pickle', 'rb') as handle:
        return pickle.load(handle)