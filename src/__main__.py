import logging
import os
from datetime import datetime

import fire
import warnings
from configue import load_config_from_file

warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)


class CLI:
    logger = logging.getLogger(__name__)

    def run_command(self, config_path: str, command_name: str):
        self._run_command(config_path, command_name)

    @staticmethod
    def _run_command(config_path, command_name):
        config = load_config_from_file(config_path)
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"forecast_{command_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        config["logging"]["handlers"]["file"]["filename"] = os.path.join(log_dir, log_filename)
        command = config[command_name]
        command.run()


if __name__ == "__main__":
    fire.Fire(CLI)
