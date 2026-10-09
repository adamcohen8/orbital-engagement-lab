//! Immutable a98cd905 scalar oracle for exact arithmetic and error compatibility.
//! Compiled only by Rust tests; production has one force implementation.
use super::{dot, finite_vector, norm, AU_KM, DAY_S, SOLAR_PRESSURE_PA};

fn reference(
    position_km: [f64; 3],
    sun_position_km: [f64; 3],
    elapsed_s: f64,
    radius_km: f64,
    nodes: &[f64],
    weights_gauss: &[f64],
    cos_azimuth: &[f64],
    sin_azimuth: &[f64],
    include_albedo: bool,
    include_infrared: bool,
    uniform_albedo: Option<f64>,
    uniform_emissivity: Option<f64>,
) -> Result<([f64; 3], [f64; 3]), String> {
    finite_vector(position_km, "spacecraft position")?;
    finite_vector(sun_position_km, "Sun position")?;
    if !elapsed_s.is_finite()
        || !radius_km.is_finite()
        || radius_km <= 0.0
        || nodes.is_empty()
        || nodes.len() != weights_gauss.len()
        || cos_azimuth.len() != 4 * nodes.len()
        || sin_azimuth.len() != cos_azimuth.len()
    {
        return Err("Earth radiation geometry or quadrature dimensions are invalid".to_owned());
    }
    if nodes
        .iter()
        .chain(weights_gauss)
        .chain(cos_azimuth)
        .chain(sin_azimuth)
        .any(|value| !value.is_finite())
    {
        return Err("Earth radiation quadrature values must be finite".to_owned());
    }
    if let (Some(albedo), Some(emissivity)) = (uniform_albedo, uniform_emissivity) {
        if !albedo.is_finite()
            || !emissivity.is_finite()
            || !(0.0..=1.0).contains(&albedo)
            || !(0.0..=1.0).contains(&emissivity)
        {
            return Err("uniform albedo and emissivity must lie in [0, 1]".to_owned());
        }
    } else if uniform_albedo.is_some() || uniform_emissivity.is_some() {
        return Err("uniform albedo and emissivity must be supplied together".to_owned());
    }
    if !include_albedo && !include_infrared {
        return Ok(([0.0; 3], [0.0; 3]));
    }
    let distance = norm(position_km);
    let sun_distance = norm(sun_position_km);
    if distance <= radius_km || sun_distance <= radius_km {
        return Err("Earth radiation requires spacecraft and Sun outside Earth".to_owned());
    }
    let radial = [
        position_km[0] / distance,
        position_km[1] / distance,
        position_km[2] / distance,
    ];
    let mut east = [-radial[1], radial[0], 0.0];
    let east_norm = norm(east);
    if east_norm < 1.0e-12 {
        east = [0.0, 1.0, 0.0];
    } else {
        east = east.map(|value| value / east_norm);
    }
    let north = [
        radial[1] * east[2] - radial[2] * east[1],
        radial[2] * east[0] - radial[0] * east[2],
        radial[0] * east[1] - radial[1] * east[0],
    ];
    let ratio = radius_km / distance;
    let cos_limb = (1.0 - ratio * ratio).max(0.0).sqrt();
    let width = ratio * ratio / (1.0 + cos_limb);
    let pressure = SOLAR_PRESSURE_PA * (AU_KM / sun_distance).powi(2);
    let sun_hat = sun_position_km.map(|value| value / sun_distance);
    let mut albedo_pressure = [0.0; 3];
    let mut infrared_pressure = [0.0; 3];
    let order = nodes.len();
    let seasonal = (2.0 * std::f64::consts::PI * elapsed_s / (365.25 * DAY_S)).cos();
    for (i, &node) in nodes.iter().enumerate() {
        let cosine = 1.0 - width * (1.0 - node) / 2.0;
        let sine = (1.0 - cosine * cosine).max(0.0).sqrt();
        let root = (radius_km * radius_km - distance * distance * (1.0 - cosine * cosine))
            .max(0.0)
            .sqrt();
        let ray_length = (distance * distance - radius_km * radius_km) / (distance * cosine + root);
        let quadrature_weight =
            weights_gauss[i] * (width / 2.0) * (2.0 * std::f64::consts::PI / (4.0 * order as f64));
        for j in 0..(4 * order) {
            let direction = [
                cosine * radial[0] + sine * (cos_azimuth[j] * east[0] + sin_azimuth[j] * north[0]),
                cosine * radial[1] + sine * (cos_azimuth[j] * east[1] + sin_azimuth[j] * north[1]),
                cosine * radial[2] + sine * (cos_azimuth[j] * east[2] + sin_azimuth[j] * north[2]),
            ];
            let normal = [
                (position_km[0] - ray_length * direction[0]) / radius_km,
                (position_km[1] - ray_length * direction[1]) / radius_km,
                (position_km[2] - ray_length * direction[2]) / radius_km,
            ];
            let (albedo, emissivity) =
                if let (Some(a), Some(e)) = (uniform_albedo, uniform_emissivity) {
                    (a, e)
                } else {
                    let latitude = normal[2];
                    let p2 = 0.5 * (3.0 * latitude * latitude - 1.0);
                    (
                        0.34 + 0.10 * seasonal * latitude + 0.29 * p2,
                        0.68 - 0.07 * seasonal * latitude - 0.18 * p2,
                    )
                };
            if include_albedo {
                let incidence = dot(normal, sun_hat).max(0.0);
                for axis in 0..3 {
                    albedo_pressure[axis] +=
                        quadrature_weight * direction[axis] * albedo * incidence * pressure
                            / std::f64::consts::PI;
                }
            }
            if include_infrared {
                for axis in 0..3 {
                    infrared_pressure[axis] +=
                        quadrature_weight * direction[axis] * emissivity * pressure
                            / (4.0 * std::f64::consts::PI);
                }
            }
        }
    }
    Ok((albedo_pressure, infrared_pressure))
}

#[derive(Clone)]
struct Case {
    position: [f64; 3],
    sun: [f64; 3],
    elapsed: f64,
    radius: f64,
    nodes: Vec<f64>,
    weights: Vec<f64>,
    cosine: Vec<f64>,
    sine: Vec<f64>,
    albedo: bool,
    infrared: bool,
    uniform_a: Option<f64>,
    uniform_e: Option<f64>,
}

impl Case {
    fn new(order: usize) -> Self {
        let azimuth: Vec<f64> = (0..4 * order)
            .map(|i| (i as f64 + 0.5) * (2.0 * std::f64::consts::PI / (4 * order) as f64))
            .collect();
        Self {
            position: [6800., 1200., -450.],
            sun: [1.3e8, 5.6e7, -4.2e6],
            elapsed: -1e7,
            radius: 6378.137,
            nodes: (0..order)
                .map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / order as f64)
                .collect(),
            weights: vec![2.0 / order as f64; order],
            cosine: azimuth.iter().map(|v| v.cos()).collect(),
            sine: azimuth.iter().map(|v| v.sin()).collect(),
            albedo: true,
            infrared: true,
            uniform_a: None,
            uniform_e: None,
        }
    }
    fn assert_exact(&self) {
        let old = reference(
            self.position,
            self.sun,
            self.elapsed,
            self.radius,
            &self.nodes,
            &self.weights,
            &self.cosine,
            &self.sine,
            self.albedo,
            self.infrared,
            self.uniform_a,
            self.uniform_e,
        );
        let new = super::earth_radiation_components(
            self.position,
            self.sun,
            self.elapsed,
            self.radius,
            &self.nodes,
            &self.weights,
            &self.cosine,
            &self.sine,
            self.albedo,
            self.infrared,
            self.uniform_a,
            self.uniform_e,
        );
        match (old, new) {
            (Ok((a, ir)), Ok((b, jr))) => {
                assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
                assert_eq!(ir.map(f64::to_bits), jr.map(f64::to_bits));
            }
            (Err(a), Err(b)) => assert_eq!(a, b),
            other => panic!("validation disposition changed: {other:?}"),
        }
    }
}

#[test]
fn optimized_earth_radiation_matches_scalar_bits_across_geometry_and_flags() {
    for order in [1, 3, 8, 9, 16, 32, 64, 128] {
        let mut case = Case::new(order);
        let geometries = [
            ([6800., 1200., -450.], -1e7),
            ([0., 0., 7100.], 0.),
            ([0., 0., -7100.], 123.25),
            ([-12000., 4000., 700.], 4e8),
            ([42164., -250., 2200.], -4e8),
            ([6378.1370001, 0., 0.], 1e8),
            ([1e-10, 0., 7000.], 1e8),
            ([1e-6, 0., -7000.], 1e8),
            ([1e200, 1e200, -1e200], 1e8),
            ([7000., 0., 0.], 1e308),
        ];
        for (position, elapsed) in geometries {
            case.position = position;
            case.elapsed = elapsed;
            for sun in [
                [1.3e8, 5.6e7, -4.2e6],
                [-1.3e8, -5.6e7, 4.2e6],
                [1e308, 1e308, 0.],
            ] {
                case.sun = sun;
                for (albedo, infrared) in
                    [(true, true), (true, false), (false, true), (false, false)]
                {
                    case.albedo = albedo;
                    case.infrared = infrared;
                    for uniform in [None, Some((0.3, 0.7)), Some((0., 1.))] {
                        (case.uniform_a, case.uniform_e) =
                            uniform.map_or((None, None), |(a, e)| (Some(a), Some(e)));
                        case.assert_exact();
                    }
                }
            }
        }
    }
}

#[test]
fn optimized_earth_radiation_preserves_validation_error_order() {
    let valid = Case::new(8);
    let mut invalid = valid.clone();
    invalid.position[0] = f64::NAN;
    invalid.radius = -1.;
    invalid.assert_exact();
    let mut invalid = valid.clone();
    invalid.sun[1] = f64::INFINITY;
    invalid.radius = -1.;
    invalid.assert_exact();
    let mut invalid = valid.clone();
    invalid.elapsed = f64::NAN;
    invalid.weights.clear();
    invalid.assert_exact();
    for radius in [-1., 0., f64::NAN, f64::INFINITY] {
        let mut invalid = valid.clone();
        invalid.radius = radius;
        invalid.assert_exact();
    }
    let mut invalid = valid.clone();
    invalid.nodes.clear();
    invalid.assert_exact();
    let mut invalid = valid.clone();
    invalid.weights.pop();
    invalid.assert_exact();
    let mut invalid = valid.clone();
    invalid.cosine.pop();
    invalid.assert_exact();
    let mut invalid = valid.clone();
    invalid.sine.pop();
    invalid.assert_exact();
    for field in 0..4 {
        let mut invalid = valid.clone();
        match field {
            0 => invalid.nodes[0] = f64::NAN,
            1 => invalid.weights[0] = f64::INFINITY,
            2 => invalid.cosine[0] = f64::NAN,
            _ => invalid.sine[0] = f64::NEG_INFINITY,
        }
        invalid.assert_exact();
    }
    for (a, e) in [
        (Some(0.3), None),
        (None, Some(0.7)),
        (Some(f64::NAN), Some(0.7)),
        (Some(-0.1), Some(0.7)),
        (Some(0.3), Some(1.1)),
    ] {
        let mut invalid = valid.clone();
        invalid.uniform_a = a;
        invalid.uniform_e = e;
        invalid.assert_exact();
    }
    let mut invalid = valid.clone();
    invalid.position = [0.; 3];
    invalid.assert_exact();
    invalid.albedo = false;
    invalid.infrared = false;
    invalid.assert_exact();
    let mut invalid = valid.clone();
    invalid.sun = [0.; 3];
    invalid.assert_exact();
}

#[test]
fn validated_batch_rows_preserve_row_validation_and_scalar_bits() {
    let case = Case::new(8);
    super::earth_radiation_components(
        case.position,
        case.sun,
        case.elapsed,
        case.radius,
        &case.nodes,
        &case.weights,
        &case.cosine,
        &case.sine,
        true,
        true,
        None,
        None,
    )
    .unwrap();
    for position in [[0., 0., 7100.], [7000., f64::NAN, 0.], [0.; 3]] {
        for elapsed in [0., f64::NAN] {
            let old = reference(
                position,
                case.sun,
                elapsed,
                case.radius,
                &case.nodes,
                &case.weights,
                &case.cosine,
                &case.sine,
                true,
                true,
                None,
                None,
            );
            let new = super::earth_radiation_components_prevalidated(
                position,
                case.sun,
                elapsed,
                case.radius,
                &case.nodes,
                &case.weights,
                &case.cosine,
                &case.sine,
                true,
                true,
                None,
                None,
            );
            match (old, new) {
                (Ok((a, ir)), Ok((b, jr))) => {
                    assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
                    assert_eq!(ir.map(f64::to_bits), jr.map(f64::to_bits));
                }
                (Err(a), Err(b)) => assert_eq!(a, b),
                other => panic!("batch row contract changed: {other:?}"),
            }
        }
    }
}
