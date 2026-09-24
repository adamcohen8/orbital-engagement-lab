"""Orbital Engineering Lab Pro campaign file alternatives are not included in the public core."""


def is_file_location_path(path: str) -> bool:
    terminal = path.rsplit(".", 1)[-1].split("[", 1)[0]
    return terminal.endswith(("_path", "_file")) or terminal in {
        "output_dir", "summary_json", "prompt_file", "spice_kernels",
    }


def is_campaign_input_file_path(path: str) -> bool:
    return False
