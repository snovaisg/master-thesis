from pydantic import BaseSettings


class Settings(BaseSettings):
    mimic_path: str = 'Define me in .env'
    ccs_path : str = 'Define me in .env'
    icd9_chapter_path : str = 'Define me in .env'

    class Config:
        env_file = ".env"
