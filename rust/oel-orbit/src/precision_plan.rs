//! Prepared precision forces for the ordered ONP plan. Physics stays with its
//! existing kernel owners; this adapter supplies immutable tables and frames.

use crate::{ocean_tides::OceanTidesContext, precision_forces, solid_tides};

pub const RADIATION: u8 = 9;
pub const SCHWARZSCHILD: u8 = 10;
pub const SOLID: u8 = 11;
pub const OCEAN: u8 = 12;
pub const STAGE_WIDTH: usize = 21;
pub type Specification = (u8, Vec<f64>, Vec<Vec<f64>>);

pub enum PrecisionForce {
    Radiation {
        area: f64,
        albedo: bool,
        infrared: bool,
        quadrature: Vec<Vec<f64>>,
    },
    Schwarzschild,
    Solid {
        zero_tide: bool,
        pole: bool,
        sun_mu: f64,
        moon_mu: f64,
    },
    Ocean {
        context: OceanTidesContext,
        pole: bool,
    },
}

pub struct PrecisionStage {
    pub rotation: [f64; 9],
    pub sun: [f64; 3],
    pub moon: [f64; 3],
    pub jd_tt: f64,
    pub jd_ut1: f64,
    pub pole: [f64; 2],
    pub elapsed: f64,
    pub radius: f64,
}

impl PrecisionStage {
    pub fn from_flat(row: &[f64]) -> Result<Self, &'static str> {
        if row.len() != STAGE_WIDTH || row.iter().any(|v| !v.is_finite()) {
            return Err("precision stage inputs must have 21 finite values");
        }
        Ok(Self {
            rotation: row[..9].try_into().unwrap(),
            sun: row[9..12].try_into().unwrap(),
            moon: row[12..15].try_into().unwrap(),
            jd_tt: row[15],
            jd_ut1: row[16],
            pole: row[17..19].try_into().unwrap(),
            elapsed: row[19],
            radius: row[20],
        })
    }
}

fn mv(rotation: &[f64; 9], vector: [f64; 3], transpose: bool) -> [f64; 3] {
    std::array::from_fn(|i| {
        let index = |j| if transpose { j * 3 + i } else { i * 3 + j };
        rotation[index(0)] * vector[0]
            + rotation[index(1)] * vector[1]
            + rotation[index(2)] * vector[2]
    })
}

impl PrecisionForce {
    pub fn new((code, values, tables): Specification) -> Result<Self, String> {
        if values
            .iter()
            .chain(tables.iter().flatten())
            .any(|v| !v.is_finite())
        {
            return Err("precision force specifications must be finite".into());
        }
        match code {
            RADIATION
                if values.len() == 3
                    && tables.len() == 4
                    && values[0] >= 0.0
                    && (8..=128).contains(&tables[0].len())
                    && tables[1].len() == tables[0].len()
                    && tables[2].len() == 4 * tables[0].len()
                    && tables[3].len() == tables[2].len()
                    && matches!(values[1], 0.0 | 1.0)
                    && matches!(values[2], 0.0 | 1.0)
                    && (values[1] == 1.0 || values[2] == 1.0) =>
            {
                Ok(Self::Radiation {
                    area: values[0],
                    albedo: values[1] == 1.0,
                    infrared: values[2] == 1.0,
                    quadrature: tables,
                })
            }
            SCHWARZSCHILD if values.is_empty() && tables.is_empty() => Ok(Self::Schwarzschild),
            SOLID
                if values.len() == 4
                    && tables.is_empty()
                    && matches!(values[0], 0.0 | 1.0)
                    && matches!(values[1], 0.0 | 1.0)
                    && values[2] > 0.0
                    && values[3] > 0.0 =>
            {
                Ok(Self::Solid {
                    zero_tide: values[0] == 1.0,
                    pole: values[1] == 1.0,
                    sun_mu: values[2],
                    moon_mu: values[3],
                })
            }
            OCEAN
                if values.len() == 2
                    && tables.len() == 2
                    && values[0].fract() == 0.0
                    && (2.0..=20.0).contains(&values[0])
                    && matches!(values[1], 0.0 | 1.0) =>
            {
                let mut tables = tables.into_iter();
                Ok(Self::Ocean {
                    context: OceanTidesContext::new(
                        tables.next().unwrap(),
                        tables.next().unwrap(),
                        values[0] as usize,
                    )?,
                    pole: values[1] == 1.0,
                })
            }
            _ => Err("invalid precision force specification".into()),
        }
    }

    pub fn code(&self) -> u8 {
        match self {
            Self::Radiation { .. } => RADIATION,
            Self::Schwarzschild => SCHWARZSCHILD,
            Self::Solid { .. } => SOLID,
            Self::Ocean { .. } => OCEAN,
        }
    }

    pub fn acceleration(
        &self,
        state: [f64; 6],
        stage: &PrecisionStage,
        mu: f64,
        mass: f64,
        reflectivity: f64,
    ) -> Result<[f64; 3], String> {
        if matches!(self, Self::Schwarzschild) {
            return precision_forces::schwarzschild_acceleration(state, mu);
        }
        if let Self::Radiation {
            area,
            albedo,
            infrared,
            quadrature: q,
        } = self
        {
            if !mass.is_finite() || mass <= 0.0 || !reflectivity.is_finite() || reflectivity < 0.0 {
                return Err(
                    "Earth radiation requires positive mass and nonnegative finite area/Cr.".into(),
                );
            }
            if *area == 0.0 || reflectivity == 0.0 {
                return Ok([0.0; 3]);
            }
            let position = mv(&stage.rotation, state[..3].try_into().unwrap(), false);
            let (a, ir) = precision_forces::earth_radiation_components(
                position,
                stage.sun,
                stage.elapsed,
                6378.137,
                &q[0],
                &q[1],
                &q[2],
                &q[3],
                *albedo,
                *infrared,
                None,
                None,
            )?;
            let scale = area * reflectivity / mass / 1000.0;
            let a = mv(&stage.rotation, a, true);
            let ir = mv(&stage.rotation, ir, true);
            return Ok(std::array::from_fn(|i| a[i] * scale + ir[i] * scale));
        }
        let position = mv(&stage.rotation, state[..3].try_into().unwrap(), false);
        let fixed = match self {
            Self::Solid {
                zero_tide,
                pole,
                sun_mu,
                moon_mu,
            } => solid_tides::acceleration(
                position,
                stage.sun,
                stage.moon,
                mu,
                stage.radius,
                stage.jd_tt,
                stage.jd_ut1,
                if *zero_tide { "zero_tide" } else { "tide_free" },
                if *pole {
                    Some((stage.pole[0], stage.pole[1]))
                } else {
                    None
                },
                *sun_mu,
                *moon_mu,
            )?,
            Self::Ocean { context, pole } => context.acceleration(
                position,
                stage.jd_tt,
                stage.jd_ut1,
                if *pole {
                    Some((stage.pole[0], stage.pole[1]))
                } else {
                    None
                },
                mu,
                stage.radius,
            )?,
            _ => unreachable!(),
        };
        Ok(mv(&stage.rotation, fixed, true))
    }
}
