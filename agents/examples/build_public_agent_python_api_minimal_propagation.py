from sim import ScenarioBuilder


def main() -> int:
    artifact = (
        ScenarioBuilder("public_agent_python_api_minimal_propagation")
        .description(
            "Expected artifact shape for the Python API minimal propagation task card, "
            "using a five-minute deterministic propagation horizon."
        )
        .duration(300.0, dt_s=10.0)
        .target_satellite(
            mass_kg=300.0,
            position_eci_km=[7000.0, 0.0, 0.0],
            velocity_eci_km_s=[0.0, 7.5, 0.0],
        )
        .outputs(
            "outputs/agents/public_agent_python_api_minimal_propagation",
            stats={"print_summary": False, "save_json": True, "save_full_log": False},
        )
        .review(detail="standard")
        .artifact()
    )

    report = artifact.validate_report()
    if not report.ok:
        raise SystemExit(report.to_dict())
    artifact.write("agents/examples/public_agent_python_api_minimal_propagation.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
