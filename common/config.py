from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gamma_api_base_url: str = "https://gamma-api.polymarket.com"
    weather_api_base_url: str = "https://wttr.in"
    weather_locations: list[str] = ["Miami", "Houston", "New York"]
    log_level: str = "INFO"

    quality_high_threshold: float = 0.02
    quality_low_threshold: float = 0.06

    data_dir: str = "."

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
