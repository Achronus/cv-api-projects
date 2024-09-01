from pathlib import Path
import re

from pydantic import BaseModel, field_validator, validate_call
from pydantic_core import PydanticCustomError

FILENAME_PATTERN = r"^[\w\s.-]{1,255}(\.[\w]{1,10})"


class StrPathValidation(BaseModel):
    value: str

    @field_validator("value")
    def validate_value(cls, value: str) -> str:
        pattern = re.compile(FILENAME_PATTERN)
        if value.startswith("http") or pattern.match(value):
            return value

        for item in ["./", "../", "/"]:
            if value.startswith(item):
                return value

        raise PydanticCustomError(
            "invalid_string",
            "Must be a valid filename format or start with any of the following: ['http', './', '../', '/']",
            dict(wrong_value=value),
        )


@validate_call(validate_return=True)
def str_to_path(value: str | Path) -> Path:
    """
    Converts a string to a Path object based on specific rules. Returns URLs as they are.

    - If a `pathlib.Path`, returns the value as is.
    - If the input starts with `http`, returns the value as is.
    - If the input starts with `./` or `../`, treat it as a relative path.
    - If the input starts with `/`, treat it as a sub-directory in the `data` folder.
    - If only a filename is provided, prepend `cwd/data/` to it.
    """
    if isinstance(value, Path):
        return value

    value = StrPathValidation(value=value).value

    if value.startswith("http"):
        return value

    base_path = Path(Path.cwd(), "data")
    path = Path(value)

    if value.startswith("./") or value.startswith("../"):
        return path.resolve()
    elif value.startswith("/"):
        return Path(base_path, path.relative_to("/"))
    else:
        return Path(base_path, value)
