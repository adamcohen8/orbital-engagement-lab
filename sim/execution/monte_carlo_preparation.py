# ruff: noqa: F401,F403,F405,I001
from .campaign_common import *

def sample_monte_carlo_variation(variation: Any, rng: np.random.Generator) -> Any:
    mode = variation.mode.lower()
    if mode == "choice":
        if not variation.options:
            raise ValueError(f"Variation '{variation.parameter_path}' with mode=choice requires options.")
        return variation.options[int(rng.integers(0, len(variation.options)))]
    if mode == "uniform":
        if variation.low is None or variation.high is None:
            raise ValueError(f"Variation '{variation.parameter_path}' with mode=uniform requires low/high.")
        return float(rng.uniform(variation.low, variation.high))
    if mode == "normal":
        if variation.mean is None or variation.std is None:
            raise ValueError(f"Variation '{variation.parameter_path}' with mode=normal requires mean/std.")
        return float(rng.normal(variation.mean, variation.std))
    raise ValueError(f"Unsupported variation mode '{variation.mode}'.")


def prepare_monte_carlo_runs(
    *,
    cfg: SimulationScenarioConfig,
    root: dict[str, Any],
    outdir: Path,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(cfg.monte_carlo.base_seed))
    varies_metadata_seed = any(str(v.parameter_path) == "metadata.seed" for v in cfg.monte_carlo.variations)
    prepared: list[dict[str, Any]] = []
    for i in range(int(cfg.monte_carlo.iterations)):
        cdict = deepcopy(root)
        sampled = {}
        for variation in cfg.monte_carlo.variations:
            sampled_value = sample_monte_carlo_variation(variation, rng)
            set_parameter_path_value(cdict, variation.parameter_path, sampled_value)
            sampled[variation.parameter_path] = sampled_value
        if not varies_metadata_seed:
            md = cdict.setdefault("metadata", {})
            md["seed"] = int(cfg.monte_carlo.base_seed) + i
        mode = str(cdict.get("outputs", {}).get("mode", "interactive"))
        if mode == "interactive":
            cdict.setdefault("outputs", {})["mode"] = "save"
        cdict.setdefault("outputs", {})["output_dir"] = str(outdir / f"mc_run_{i:04d}")
        config_hash = _config_fingerprint(cdict)
        prepared.append(
            {
                "iteration": i,
                "sampled_parameters": sampled,
                "config_dict": cdict,
                "config_hash": config_hash,
                "seed": int(cdict.get("metadata", {}).get("seed", int(cfg.monte_carlo.base_seed) + i)),
            }
        )
    return prepared

__all__ = [name for name in globals() if not name.startswith("__")]
