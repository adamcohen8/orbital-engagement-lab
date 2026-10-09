//! IERS 2010 solid Earth tidal coefficients and fused tidal acceleration.
//!
//! The scalar order, Love numbers, frequency terms and mean-pole model match
//! OEL's Python reference. Time scales, EOP and body positions remain caller owned.
//! Frequency tables 6.5a/b/c are transcribed from Orekit 13.1.7 (Apache-2.0);
//! see NOTICE.txt for attribution.

use crate::perturbations::{solid_harmonics_values, tidal_acceleration_fixed};

const ARCSEC: f64 = std::f64::consts::PI / 648000.0;
const LOVE: [(usize, usize, f64, f64, f64); 7] = [
    (2, 0, 0.30190, 0.0, -0.00089),
    (2, 1, 0.29830, -0.00144, -0.00080),
    (2, 2, 0.30102, -0.00130, -0.00057),
    (3, 0, 0.093, 0.0, 0.0),
    (3, 1, 0.093, 0.0, 0.0),
    (3, 2, 0.093, 0.0, 0.0),
    (3, 3, 0.094, 0.0, 0.0),
];
const DELAUNAY: [[f64; 5]; 5] = [
    [
        134.96340251 * 3600.0,
        1717915923.2178,
        31.8792,
        0.051635,
        -0.00024470,
    ],
    [
        357.52910918 * 3600.0,
        129596581.0481,
        -0.5532,
        0.000136,
        -0.00001149,
    ],
    [
        93.27209062 * 3600.0,
        1739527262.8478,
        -12.7512,
        -0.001037,
        0.00000417,
    ],
    [
        297.85019547 * 3600.0,
        1602961601.2090,
        -6.3706,
        0.006593,
        -0.00003169,
    ],
    [
        125.04455501 * 3600.0,
        -6962890.5431,
        7.4722,
        0.007702,
        -0.00005939,
    ],
];
type FrequencyTerm = (f64, [f64; 5], f64, f64);
const K20: [FrequencyTerm; 21] = [
    (0.0, [0.0, 0.0, 0.0, 0.0, 1.0], 16.6, -6.7),
    (0.0, [0.0, 0.0, 0.0, 0.0, 2.0], -0.1, 0.1),
    (0.0, [0.0, -1.0, 0.0, 0.0, 0.0], -1.2, 0.8),
    (0.0, [0.0, 0.0, -2.0, 2.0, -2.0], -5.5, 4.3),
    (0.0, [0.0, 0.0, -2.0, 2.0, -1.0], 0.1, -0.1),
    (0.0, [0.0, -1.0, -2.0, 2.0, -2.0], -0.3, 0.2),
    (0.0, [1.0, 0.0, 0.0, -2.0, 0.0], -0.3, 0.7),
    (0.0, [-1.0, 0.0, 0.0, 0.0, -1.0], 0.1, -0.2),
    (0.0, [-1.0, 0.0, 0.0, 0.0, 0.0], -1.2, 3.7),
    (0.0, [-1.0, 0.0, 0.0, 0.0, 1.0], 0.1, -0.2),
    (0.0, [1.0, 0.0, -2.0, 0.0, -2.0], 0.1, -0.2),
    (0.0, [0.0, 0.0, 0.0, -2.0, 0.0], 0.0, 0.6),
    (0.0, [-2.0, 0.0, 0.0, 0.0, 0.0], 0.0, 0.3),
    (0.0, [0.0, 0.0, -2.0, 0.0, -2.0], 0.6, 6.3),
    (0.0, [0.0, 0.0, -2.0, 0.0, -1.0], 0.2, 2.6),
    (0.0, [0.0, 0.0, -2.0, 0.0, 0.0], 0.0, 0.2),
    (0.0, [1.0, 0.0, -2.0, -2.0, -2.0], 0.1, 0.2),
    (0.0, [-1.0, 0.0, -2.0, 0.0, -2.0], 0.4, 1.1),
    (0.0, [-1.0, 0.0, -2.0, 0.0, -1.0], 0.2, 0.5),
    (0.0, [0.0, 0.0, -2.0, -2.0, -2.0], 0.1, 0.2),
    (0.0, [-2.0, 0.0, -2.0, 0.0, -2.0], 0.1, 0.1),
];
const K21: [FrequencyTerm; 48] = [
    (1.0, [2.0, 0.0, 2.0, 0.0, 2.0], -0.1, 0.0),
    (1.0, [0.0, 0.0, 2.0, 2.0, 2.0], -0.1, 0.0),
    (1.0, [1.0, 0.0, 2.0, 0.0, 1.0], -0.1, 0.0),
    (1.0, [1.0, 0.0, 2.0, 0.0, 2.0], -0.7, 0.1),
    (1.0, [-1.0, 0.0, 2.0, 2.0, 2.0], -0.1, 0.0),
    (1.0, [0.0, 0.0, 2.0, 0.0, 1.0], -1.3, 0.1),
    (1.0, [0.0, 0.0, 2.0, 0.0, 2.0], -6.8, 0.6),
    (1.0, [0.0, 0.0, 0.0, 2.0, 0.0], 0.1, 0.0),
    (1.0, [1.0, 0.0, 2.0, -2.0, 2.0], 0.1, 0.0),
    (1.0, [-1.0, 0.0, 2.0, 0.0, 1.0], 0.1, 0.0),
    (1.0, [-1.0, 0.0, 2.0, 0.0, 2.0], 0.4, 0.0),
    (1.0, [1.0, 0.0, 0.0, 0.0, 0.0], 1.3, -0.1),
    (1.0, [1.0, 0.0, 0.0, 0.0, 1.0], 0.3, 0.0),
    (1.0, [-1.0, 0.0, 0.0, 2.0, 0.0], 0.3, 0.0),
    (1.0, [-1.0, 0.0, 0.0, 2.0, 1.0], 0.1, 0.0),
    (1.0, [0.0, 1.0, 2.0, -2.0, 2.0], -1.9, 0.1),
    (1.0, [0.0, 0.0, 2.0, -2.0, 1.0], 0.5, 0.0),
    (1.0, [0.0, 0.0, 2.0, -2.0, 2.0], -43.4, 2.9),
    (1.0, [0.0, -1.0, 2.0, -2.0, 2.0], 0.6, 0.0),
    (1.0, [0.0, 1.0, 0.0, 0.0, 0.0], 1.6, -0.1),
    (1.0, [-2.0, 0.0, 2.0, 0.0, 1.0], 0.1, 0.0),
    (1.0, [0.0, 0.0, 0.0, 0.0, -2.0], 0.1, 0.0),
    (1.0, [0.0, 0.0, 0.0, 0.0, -1.0], -8.8, 0.5),
    (1.0, [0.0, 0.0, 0.0, 0.0, 0.0], 470.9, -30.2),
    (1.0, [0.0, 0.0, 0.0, 0.0, 1.0], 68.1, -4.6),
    (1.0, [0.0, 0.0, 0.0, 0.0, 2.0], -1.6, 0.1),
    (1.0, [-1.0, 0.0, 0.0, 1.0, 0.0], 0.1, 0.0),
    (1.0, [0.0, -1.0, 0.0, 0.0, -1.0], -0.1, 0.0),
    (1.0, [0.0, -1.0, 0.0, 0.0, 0.0], -20.6, -0.3),
    (1.0, [0.0, 1.0, -2.0, 2.0, -2.0], 0.3, 0.0),
    (1.0, [0.0, -1.0, 0.0, 0.0, 1.0], -0.3, 0.0),
    (1.0, [-2.0, 0.0, 0.0, 2.0, 0.0], -0.2, 0.0),
    (1.0, [-2.0, 0.0, 0.0, 2.0, 1.0], -0.1, 0.0),
    (1.0, [0.0, 0.0, -2.0, 2.0, -2.0], -5.0, 0.3),
    (1.0, [0.0, 0.0, -2.0, 2.0, -1.0], 0.2, 0.0),
    (1.0, [0.0, -1.0, -2.0, 2.0, -2.0], -0.2, 0.0),
    (1.0, [1.0, 0.0, 0.0, -2.0, 0.0], -0.5, 0.0),
    (1.0, [1.0, 0.0, 0.0, -2.0, 1.0], -0.1, 0.0),
    (1.0, [-1.0, 0.0, 0.0, 0.0, -1.0], 0.1, 0.0),
    (1.0, [-1.0, 0.0, 0.0, 0.0, 0.0], -2.1, 0.1),
    (1.0, [-1.0, 0.0, 0.0, 0.0, 1.0], -0.4, 0.0),
    (1.0, [0.0, 0.0, 0.0, -2.0, 0.0], -0.2, 0.0),
    (1.0, [-2.0, 0.0, 0.0, 0.0, 0.0], -0.1, 0.0),
    (1.0, [0.0, 0.0, -2.0, 0.0, -2.0], -0.6, 0.0),
    (1.0, [0.0, 0.0, -2.0, 0.0, -1.0], -0.4, 0.0),
    (1.0, [0.0, 0.0, -2.0, 0.0, 0.0], -0.1, 0.0),
    (1.0, [-1.0, 0.0, -2.0, 0.0, -2.0], -0.1, 0.0),
    (1.0, [-1.0, 0.0, -2.0, 0.0, -1.0], -0.1, 0.0),
];
const K22: [FrequencyTerm; 2] = [
    (2.0, [1.0, 0.0, 2.0, 0.0, 2.0], -0.3, 0.0),
    (2.0, [0.0, 0.0, 2.0, 0.0, 2.0], -1.2, 0.0),
];

fn poly(coefficients: &[f64], x: f64) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |result, coefficient| result * x + coefficient)
}

fn body_distance(body: [f64; 3], label: &str, radius_km: f64) -> Result<f64, String> {
    let distance = (body[0] * body[0] + body[1] * body[1] + body[2] * body[2]).sqrt();
    if !body.iter().all(|x| x.is_finite()) || distance == 0.0 {
        return Err(format!("{label} must be a finite nonzero 3-vector."));
    }
    if distance <= radius_km {
        return Err(format!("{label} must lie outside Earth for solid tides."));
    }
    Ok(distance)
}

/// Return additive fully normalized C/S arrays in row-major 5x5 order.
#[allow(clippy::too_many_arguments)]
pub fn coefficients(
    sun_fixed: [f64; 3],
    moon_fixed: [f64; 3],
    mu: f64,
    radius_km: f64,
    jd_tt: f64,
    jd_ut1: f64,
    tide_system: &str,
    pole_xy_arcsec: Option<(f64, f64)>,
    sun_mu: f64,
    moon_mu: f64,
) -> Result<([f64; 25], [f64; 25]), String> {
    if tide_system != "tide_free" && tide_system != "zero_tide" {
        return Err("solid_earth_tides.tide_system must be tide_free or zero_tide.".to_owned());
    }
    if ![mu, radius_km, sun_mu, moon_mu]
        .iter()
        .all(|x| x.is_finite() && *x > 0.0)
    {
        return Err(
            "Tidal radii and gravitational parameters must be positive and finite.".to_owned(),
        );
    }
    if !jd_tt.is_finite() || !jd_ut1.is_finite() {
        return Err("Tidal epochs must be finite.".to_owned());
    }
    let mut c = [0.0; 25];
    let mut s = [0.0; 25];
    for (label, body, gm) in [("Sun", sun_fixed, sun_mu), ("Moon", moon_fixed, moon_mu)] {
        let distance = body_distance(body, label, radius_km)?;
        let h = solid_harmonics_values(body.map(|x| x / distance), 4);
        for (n, m, kr, ki, kp) in LOVE {
            // Python's scalar ** uses libm pow for integer powers, too.
            let scale = gm / mu * (radius_km / distance).powf((n + 1) as f64) / (2 * n + 1) as f64;
            let (re, im) = h[n * 5 + m];
            let qr = scale * re;
            let qi = scale * im;
            c[n * 5 + m] += kr * qr + ki * qi;
            s[n * 5 + m] += kr * qi - ki * qr;
            if n == 2 {
                c[20 + m] += kp * qr;
                s[20 + m] += kp * qi;
            }
        }
    }
    let t = (jd_tt - 2451545.0) / 36525.0;
    let delaunay = DELAUNAY.map(|row| poly(&row, t) * ARCSEC);
    let era =
        2.0 * std::f64::consts::PI * (0.7790572732640 + 1.00273781191135448 * (jd_ut1 - 2451545.0));
    let gamma = era
        + poly(
            &[
                0.014506,
                4612.156534,
                1.3915817,
                -0.00000044,
                -0.000029956,
                -0.0000000368,
            ],
            t,
        ) * ARCSEC
        + std::f64::consts::PI;
    for (order, table) in [
        (0, K20.as_slice()),
        (1, K21.as_slice()),
        (2, K22.as_slice()),
    ] {
        for (multiplier, arguments, ip, op) in table {
            let argument = arguments
                .iter()
                .zip(delaunay)
                .fold(0.0, |sum, (a, d)| sum + a * d);
            let phase = multiplier * gamma - argument;
            let sn = phase.sin();
            let cs = phase.cos();
            if order == 0 {
                c[10] += 1e-12 * (ip * cs - op * sn);
            } else if order == 1 {
                c[11] += 1e-12 * (ip * sn + op * cs);
                s[11] += 1e-12 * (ip * cs - op * sn);
            } else {
                c[12] += 1e-12 * ip * cs;
                s[12] -= 1e-12 * ip * sn;
            }
        }
    }
    if tide_system == "zero_tide" {
        c[10] -= 4.4228e-8 * -0.31460 * 0.30190;
    }
    if let Some((xp, yp)) = pole_xy_arcsec {
        if !xp.is_finite() || !yp.is_finite() {
            return Err("Pole coordinates must be finite arcseconds.".to_owned());
        }
        let years = (jd_tt - 2451545.0) / 365.25;
        let (mean_x, mean_y) = if jd_tt <= 2455197.5 {
            (
                poly(&[55.974, 1.8243, 0.18413, 0.007024], years) / 1000.0,
                poly(&[346.346, 1.7896, -0.10729, -0.000908], years) / 1000.0,
            )
        } else {
            (
                (23.513 + 7.6141 * years) / 1000.0,
                (358.891 - 0.6287 * years) / 1000.0,
            )
        };
        let m1 = xp - mean_x;
        let m2 = mean_y - yp;
        c[11] -= 1.333e-9 * (m1 + 0.0115 * m2);
        s[11] -= 1.333e-9 * (m2 - 0.0115 * m1);
    }
    Ok((c, s))
}

/// Generate solid-tide coefficients and evaluate their Cartesian gradient.
#[allow(clippy::too_many_arguments)]
pub fn acceleration(
    position: [f64; 3],
    sun_fixed: [f64; 3],
    moon_fixed: [f64; 3],
    mu: f64,
    radius_km: f64,
    jd_tt: f64,
    jd_ut1: f64,
    tide_system: &str,
    pole_xy_arcsec: Option<(f64, f64)>,
    sun_mu: f64,
    moon_mu: f64,
) -> Result<[f64; 3], String> {
    let (c, s) = coefficients(
        sun_fixed,
        moon_fixed,
        mu,
        radius_km,
        jd_tt,
        jd_ut1,
        tide_system,
        pole_xy_arcsec,
        sun_mu,
        moon_mu,
    )?;
    tidal_acceleration_fixed(position, &c, &s, 4, mu, radius_km)
}

#[cfg(feature = "python")]
pub fn register_py(module: &pyo3::prelude::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    #[pyfunction]
    #[pyo3(signature = (sun_fixed, moon_fixed, mu, radius_km, jd_tt, jd_ut1, tide_system, pole_xy_arcsec, sun_mu, moon_mu))]
    #[allow(clippy::too_many_arguments)]
    fn perturbation_solid_tides_coefficients(
        sun_fixed: [f64; 3],
        moon_fixed: [f64; 3],
        mu: f64,
        radius_km: f64,
        jd_tt: f64,
        jd_ut1: f64,
        tide_system: &str,
        pole_xy_arcsec: Option<(f64, f64)>,
        sun_mu: f64,
        moon_mu: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let (c, s) = coefficients(
            sun_fixed,
            moon_fixed,
            mu,
            radius_km,
            jd_tt,
            jd_ut1,
            tide_system,
            pole_xy_arcsec,
            sun_mu,
            moon_mu,
        )
        .map_err(PyValueError::new_err)?;
        Ok((c.to_vec(), s.to_vec()))
    }

    #[pyfunction]
    #[pyo3(signature = (position, sun_fixed, moon_fixed, mu, radius_km, jd_tt, jd_ut1, tide_system, pole_xy_arcsec, sun_mu, moon_mu))]
    #[allow(clippy::too_many_arguments)]
    fn perturbation_solid_tides(
        position: [f64; 3],
        sun_fixed: [f64; 3],
        moon_fixed: [f64; 3],
        mu: f64,
        radius_km: f64,
        jd_tt: f64,
        jd_ut1: f64,
        tide_system: &str,
        pole_xy_arcsec: Option<(f64, f64)>,
        sun_mu: f64,
        moon_mu: f64,
    ) -> PyResult<[f64; 3]> {
        acceleration(
            position,
            sun_fixed,
            moon_fixed,
            mu,
            radius_km,
            jd_tt,
            jd_ut1,
            tide_system,
            pole_xy_arcsec,
            sun_mu,
            moon_mu,
        )
        .map_err(PyValueError::new_err)
    }

    module.add_function(wrap_pyfunction!(
        perturbation_solid_tides_coefficients,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(perturbation_solid_tides, module)?)?;
    Ok(())
}
