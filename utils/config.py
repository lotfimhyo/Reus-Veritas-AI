from pydantic import BaseSettings

class Settings(BaseSettings):
    ENV: str = 'dev'
    API_PORT: int = 8080

settings = Settings()
