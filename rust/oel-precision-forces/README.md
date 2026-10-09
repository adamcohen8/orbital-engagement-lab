# Public OEL Earth precision-force kernels

This experimental Rust library exposes `earth_radiation`, `schwarzschild`,
`solid_earth_tides` and `ocean_tides` and their shared numeric helpers. It uses
the same authoritative source files as OEL's native implementation. The public
repository includes those owners; keep the checked-in directory layout when
building. The crate is a repository library and is not configured for Cargo
registry publication.

```sh
cargo test --release --manifest-path rust/oel-precision-forces/Cargo.toml
```

The default build needs no external tide datasets or Python runtime. It exposes
`perturbations::earth_radiation_components`,
`perturbations::schwarzschild_acceleration`,
`solid_tides::coefficients`, `solid_tides::acceleration`, and
`ocean_tides::OceanTidesContext`. Shared harmonic inputs are fully normalized
row-major C/S arrays. Inputs and accelerations use km, km/s and km/s²; radiation
components are pressures in N/m² before the caller's area/mass conversion.

Callers supply the validated quadrature, spacecraft/body positions, TT and UT1,
EOP pole coordinates, tide system and authorized numeric ocean coefficients.
No resource loader, frame transform, propagator or temporal interpolation is
implemented here. Use [OEL's configuration guide](../../docs/orbit-precision-forces.md)
for scientific conventions, optional Python/native routing and the synthetic
scenario. Optional `python` enables the existing tide binding registrations
for downstream PyO3 modules; this crate does not replace the complete OEL orbit
extension.

See [attribution](../oel-orbit/NOTICE.txt),
[Orekit's Apache license](../../sim/pro_perturbations/LICENSE-Orekit.txt), and
[the repository license](../../LICENSE.txt). External FES datasets and private
qualification fixtures are not included.
