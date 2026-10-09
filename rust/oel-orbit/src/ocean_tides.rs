//! Prepared FES normalized ocean tide coefficients and additive acceleration.
//!
//! Python validates/loads the file and supplies copied numeric arrays once.
//! This immutable context retains the existing IERS TT/UT1 phases, mean-pole
//! convention, wave ordering, and normalized solid-harmonic gradient.

use crate::perturbations::tidal_acceleration_fixed;

const ARCSEC: f64 = std::f64::consts::PI / 648000.0;
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

fn poly(coefficients: &[f64], x: f64) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |result, coefficient| result * x + coefficient)
}

/// A file-independent, immutable copy of one validated FES coefficient table.
pub struct OceanTidesContext {
    degree: usize,
    factors: Vec<[f64; 6]>,
    // Wave-major (C+ + C-, S+ + S-, S+ - S-, C+ - C-). Combining
    // these time-independent terms once preserves each original operation.
    combined: Vec<[f64; 4]>,
}

impl OceanTidesContext {
    pub fn new(factors: Vec<f64>, coefficients: Vec<f64>, degree: usize) -> Result<Self, String> {
        if !(2..=20).contains(&degree) {
            return Err("ocean tides require integer 2 <= degree <= 20".to_owned());
        }
        let count = (degree + 1) * (degree + 1);
        if factors.is_empty()
            || factors.len() % 6 != 0
            || coefficients.len() != factors.len() / 6 * count * 4
        {
            return Err("ocean tide factor/coefficient dimensions are invalid".to_owned());
        }
        if !factors
            .iter()
            .chain(coefficients.iter())
            .all(|value| value.is_finite())
        {
            return Err("ocean tide factors and coefficients must be finite".to_owned());
        }
        let combined: Vec<[f64; 4]> = coefficients
            .chunks_exact(4)
            .map(|row| {
                [
                    row[0] + row[2],
                    row[1] + row[3],
                    row[1] - row[3],
                    row[0] - row[2],
                ]
            })
            .collect();
        if !combined.iter().flatten().all(|value| value.is_finite()) {
            return Err("combined ocean tide coefficients must be finite".to_owned());
        }
        Ok(Self {
            degree,
            factors: factors
                .chunks_exact(6)
                .map(|row| row.try_into().unwrap())
                .collect(),
            combined,
        })
    }

    fn coefficients_into(
        &self,
        jd_tt: f64,
        jd_ut1: f64,
        pole_xy_arcsec: Option<(f64, f64)>,
        c: &mut [f64],
        s: &mut [f64],
    ) -> Result<(), String> {
        if !jd_tt.is_finite() || !jd_ut1.is_finite() {
            return Err("Ocean tides require finite TT and UT1 epochs.".to_owned());
        }
        if let Some((xp, yp)) = pole_xy_arcsec {
            if !xp.is_finite() || !yp.is_finite() {
                return Err(
                    "Ocean pole tide requires finite TT epoch and pole coordinates.".to_owned(),
                );
            }
        }
        let t = (jd_tt - 2451545.0) / 36525.0;
        let mut arguments = [0.0; 6];
        for (argument, coefficients) in arguments[1..].iter_mut().zip(DELAUNAY.iter()) {
            *argument = poly(coefficients, t) * ARCSEC;
        }
        let mut gamma = 2.0
            * std::f64::consts::PI
            * (0.7790572732640 + 1.00273781191135448 * (jd_ut1 - 2451545.0));
        gamma += poly(
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
        arguments[0] = gamma;
        let count = (self.degree + 1) * (self.degree + 1);
        for (wave, factors) in self.factors.iter().enumerate() {
            let mut phase = 0.0;
            for i in 0..6 {
                phase += factors[i] * arguments[i];
            }
            if !phase.is_finite() {
                return Err("ocean tide phase must be finite".to_owned());
            }
            let cosine = phase.cos();
            let sine = phase.sin();
            for (index, row) in self.combined[wave * count..(wave + 1) * count]
                .iter()
                .enumerate()
            {
                c[index] += row[0] * cosine + row[1] * sine;
                s[index] += row[2] * cosine - row[3] * sine;
            }
        }
        if let Some((xp, yp)) = pole_xy_arcsec {
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
            c[2 * (self.degree + 1) + 1] += -2.1778e-10 * (m1 - 0.01724 * m2);
            s[2 * (self.degree + 1) + 1] += -1.7232e-10 * (m2 - 0.03365 * m1);
        }
        Ok(())
    }

    pub fn coefficients(
        &self,
        jd_tt: f64,
        jd_ut1: f64,
        pole_xy_arcsec: Option<(f64, f64)>,
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let count = (self.degree + 1) * (self.degree + 1);
        let mut c = vec![0.0; count];
        let mut s = vec![0.0; count];
        self.coefficients_into(jd_tt, jd_ut1, pole_xy_arcsec, &mut c, &mut s)?;
        Ok((c, s))
    }

    pub fn acceleration(
        &self,
        position_fixed_km: [f64; 3],
        jd_tt: f64,
        jd_ut1: f64,
        pole_xy_arcsec: Option<(f64, f64)>,
        mu_km3_s2: f64,
        radius_km: f64,
    ) -> Result<[f64; 3], String> {
        // Supported degrees have bounded coefficient scratch, no per-call
        // coefficient copying through Python, and one native gradient dispatch.
        let mut c = [0.0; 441];
        let mut s = [0.0; 441];
        let count = (self.degree + 1) * (self.degree + 1);
        self.coefficients_into(
            jd_tt,
            jd_ut1,
            pole_xy_arcsec,
            &mut c[..count],
            &mut s[..count],
        )?;
        tidal_acceleration_fixed(
            position_fixed_km,
            &c[..count],
            &s[..count],
            self.degree,
            mu_km3_s2,
            radius_km,
        )
    }
}

#[cfg(feature = "python")]
mod python {
    use super::OceanTidesContext;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    #[pyclass(name = "OceanTidesContext", frozen, module = "oel_rust_orbit")]
    struct PyOceanTidesContext(OceanTidesContext);

    #[pymethods]
    impl PyOceanTidesContext {
        #[new]
        fn new(factors: Vec<f64>, coefficients: Vec<f64>, degree: usize) -> PyResult<Self> {
            OceanTidesContext::new(factors, coefficients, degree)
                .map(Self)
                .map_err(PyValueError::new_err)
        }

        #[pyo3(signature = (jd_tt, jd_ut1, pole_xy_arcsec))]
        fn coefficients(
            &self,
            jd_tt: f64,
            jd_ut1: f64,
            pole_xy_arcsec: Option<(f64, f64)>,
        ) -> PyResult<(Vec<f64>, Vec<f64>)> {
            self.0
                .coefficients(jd_tt, jd_ut1, pole_xy_arcsec)
                .map_err(PyValueError::new_err)
        }

        #[pyo3(signature = (position_fixed_km, jd_tt, jd_ut1, pole_xy_arcsec, mu_km3_s2, radius_km))]
        fn acceleration(
            &self,
            position_fixed_km: [f64; 3],
            jd_tt: f64,
            jd_ut1: f64,
            pole_xy_arcsec: Option<(f64, f64)>,
            mu_km3_s2: f64,
            radius_km: f64,
        ) -> PyResult<[f64; 3]> {
            self.0
                .acceleration(
                    position_fixed_km,
                    jd_tt,
                    jd_ut1,
                    pole_xy_arcsec,
                    mu_km3_s2,
                    radius_km,
                )
                .map_err(PyValueError::new_err)
        }
    }

    pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_class::<PyOceanTidesContext>()?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub fn register_py(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    python::register(module)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_phase_wave_retains_coefficient_arithmetic() {
        let degree = 6;
        let count = 49;
        let mut rows = vec![0.0; count * 4];
        let i = 2 * 7 + 1;
        rows[4 * i..4 * i + 4].copy_from_slice(&[1e-11, 2e-11, 3e-11, 4e-11]);
        let context = OceanTidesContext::new(vec![0.0; 6], rows, degree).unwrap();
        let (c, s) = context.coefficients(2459669.5, 2459669.5, None).unwrap();
        assert_eq!(c[i], 1e-11 + 3e-11);
        assert_eq!(s[i], 2e-11 - 4e-11);
        assert_eq!(c.iter().filter(|v| **v != 0.0).count(), 1);
        let fixed = tidal_acceleration_fixed(
            [6500.0, 2000.0, 1000.0],
            &c,
            &s,
            degree,
            398600.4418,
            6378.137,
        )
        .unwrap();
        assert_eq!(
            context
                .acceleration(
                    [6500.0, 2000.0, 1000.0],
                    2459669.5,
                    2459669.5,
                    None,
                    398600.4418,
                    6378.137
                )
                .unwrap(),
            fixed
        );
    }

    #[test]
    fn pole_zero_at_conventional_mean_before_and_after_boundary() {
        for jd in [2455197.5, 2455197.5001, 2459669.5] {
            let years = (jd - 2451545.0) / 365.25;
            let xy = if jd <= 2455197.5 {
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
            let context = OceanTidesContext::new(vec![0.0; 6], vec![0.0; 36], 2).unwrap();
            let (c, s) = context.coefficients(jd, jd, Some(xy)).unwrap();
            assert!(c.iter().chain(s.iter()).all(|value| *value == 0.0));
        }
    }

    #[test]
    fn malformed_and_nonfinite_inputs_fail_closed() {
        for degree in [0, 1, 21, usize::MAX] {
            assert!(OceanTidesContext::new(vec![0.0; 6], vec![0.0; 36], degree).is_err());
        }
        assert!(OceanTidesContext::new(vec![], vec![], 2).is_err());
        assert!(OceanTidesContext::new(vec![0.0; 5], vec![0.0; 36], 2).is_err());
        assert!(OceanTidesContext::new(vec![0.0; 6], vec![0.0; 35], 2).is_err());
        assert!(OceanTidesContext::new(vec![f64::NAN; 6], vec![0.0; 36], 2).is_err());
        assert!(OceanTidesContext::new(vec![0.0; 6], vec![f64::INFINITY; 36], 2).is_err());
        let context = OceanTidesContext::new(vec![0.0; 6], vec![0.0; 36], 2).unwrap();
        for (tt, ut1, pole) in [
            (f64::NAN, 2459669.5, None),
            (2459669.5, f64::INFINITY, None),
            (2459669.5, 2459669.5, Some((f64::NAN, 0.0))),
        ] {
            assert!(context.coefficients(tt, ut1, pole).is_err());
        }
        for (position, mu, radius) in [
            ([0.0; 3], 398600.4418, 6378.137),
            ([f64::NAN; 3], 398600.4418, 6378.137),
            ([7000.0, 0.0, 0.0], 0.0, 6378.137),
            ([7000.0, 0.0, 0.0], 398600.4418, f64::INFINITY),
        ] {
            assert!(context
                .acceleration(position, 2459669.5, 2459669.5, None, mu, radius)
                .is_err());
        }
    }
}
