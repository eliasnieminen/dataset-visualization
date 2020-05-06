import json
from typing import Optional
import pathlib


class DPDEnvironment:

    def __init__(self, settings_file: Optional[str] = "env.json") -> None:

        current_dir = pathlib.Path.cwd()
        self.environment \
            = json.loads(open(settings_file, 'r').read())["environment"]

    def get_project_root_dir(self) -> str:
        return self.environment["project_root_dir"]

    def get_audio_dir(self) -> str:
        return self.environment["audio_dir"]

    def get_unsplit_data_dir(self) -> str:
        return self.environment["unsplit_data_dir"]

    def get_serialization_dir(self) -> str:
        return self.environment["serialization_dir"]

    def get_tex_dir(self):
        return self.environment["tex_directory"]