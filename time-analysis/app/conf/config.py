from pathlib import Path
from typing import Annotated, Any

from app.conf.validators import str_to_path

from pydantic import BaseModel, BeforeValidator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class ModelDetails(BaseModel):
    ID: str
    PATH: str | None = None
    CLASSES: list[int] = []

    def model_post_init(self, __context: Any) -> None:
        base_path = Path(Path.cwd(), "models")
        self.PATH = self.PATH if self.PATH else Path(base_path, self.ID)


class Thresholds(BaseModel):
    CONFIDENCE: float = 0.5
    IOU: float = 0.5


class VideoSpecs(BaseModel):
    SRC_FILE: Annotated[str | Path, BeforeValidator(str_to_path)] = ""
    SPEED: float = 1.0
    WIDTH: int = 1024
    HEIGHT: int = 640


class Settings(BaseSettings):
    MODEL: ModelDetails
    THRESHOLD: Thresholds
    VIDEO: VideoSpecs

    model_config = SettingsConfigDict(yaml_file="config.yaml")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: BaseSettings,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            YamlConfigSettingsSource(
                settings_cls,
            ),
        )


SETTINGS = Settings()
