""" Imports """

import os
import pickle

""" Helper functions """

def save_tensor(koopman_tensor, env_id, saved_model_name):
    # Define the directory where the pickle file will be saved
    directory = f'./koopman_tensor/saved_models/{env_id}/'

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the pickle file
    with open(os.path.join(directory, f'{saved_model_name}.pickle'), 'wb') as handle:
        pickle.dump(koopman_tensor, handle)

def load_tensor(env_id, saved_model_name):
    # Define the file path
    file_path = f'./koopman_tensor/saved_models/{env_id}/{saved_model_name}.pickle'

    # Handle the case where the file does not exist
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    # Return the file if it exists
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)