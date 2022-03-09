from pydantic import BaseSettings


class Settings(BaseSettings):
    
    mimic_path: str = 'Define me in .env'
    ccs_path : str = 'Define me in .env'
    icd9_chapter_path : str = 'Define me in .env'
    random_seed : int = 'Define me in .env'
    data_base : str = 'Define me in .env' # path to save all intermediary data and results
    model_ready_dataset_folder : str = 'Define me in .env' # will be under <data_path>
    eligible_patients_folder : str = 'Define me in .env'# will be under <data_path>
    models_folder : str = 'Define me in .env'# will be under <data_path>
    variational_data_folder : str = 'Define me in .env'# will be under <data_path>
    deterministic_data_folder : str = 'Define me in .env'# will be under <data_path>
    

    class Config:
        env_file = ".env"
