//! Synthetic public reference cases; no external tide datasets or fixtures.
//! Expected values use OEL's public Python IERS 2010 coefficient/gradient owner
//! with the explicit constants below. Tolerances admit only double roundoff.

use oel_precision_forces::solid_tides::{acceleration, coefficients};

const SUN: [f64; 3] = [1.4e8, 2e7, 3e7];
const MOON: [f64; 3] = [3e5, 2e5, 1e5];
const MU: f64 = 398600.4418;
const RADIUS: f64 = 6378.137;
const SUN_MU: f64 = 132712440041.279419;
const MOON_MU: f64 = 4902.800118457551;

fn coeffs(epoch: f64, system: &str, pole: Option<(f64, f64)>) -> ([f64; 25], [f64; 25]) {
    coefficients(
        SUN,
        MOON,
        MU,
        RADIUS,
        epoch,
        epoch - 0.0008,
        system,
        pole,
        SUN_MU,
        MOON_MU,
    )
    .unwrap()
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-22 + 2e-13 * expected.abs(),
        "{actual:e} != {expected:e}"
    );
}

#[test]
fn public_python_reference_force_at_equator_both_poles_and_geo() {
    let cases = [
        (
            [7000.0, 0.0, 0.0],
            [
                -3.3090273296580364e-10,
                1.845678213399743e-10,
                1.164655621276955e-10,
            ],
        ),
        (
            [0.0, 0.0, 7000.0],
            [
                1.1439229778329176e-10,
                6.435909362178958e-11,
                2.2222253102288773e-10,
            ],
        ),
        (
            [0.0, 0.0, -7000.0],
            [
                -1.1531297566659139e-10,
                -6.497201686035475e-11,
                -2.2054164306301397e-10,
            ],
        ),
        (
            [4000.0, -5000.0, 3000.0],
            [
                2.0790504699557637e-10,
                -3.374056399996999e-11,
                8.289867488815298e-11,
            ],
        ),
        (
            [42164.0, 0.0, 0.0],
            [
                -2.5054207534629666e-13,
                1.3935143517776346e-13,
                8.779725144440298e-14,
            ],
        ),
    ];
    for (position, expected) in cases {
        let actual = acceleration(
            position,
            SUN,
            MOON,
            MU,
            RADIUS,
            2459669.5,
            2459669.4992,
            "tide_free",
            Some((0.1, 0.2)),
            SUN_MU,
            MOON_MU,
        )
        .unwrap();
        for axis in 0..3 {
            close(actual[axis], expected[axis]);
        }
    }
}

#[test]
fn zero_tide_changes_only_permanent_c20() {
    let (free_c, free_s) = coeffs(2459669.5, "tide_free", None);
    let (zero_c, zero_s) = coeffs(2459669.5, "zero_tide", None);
    close(zero_c[10] - free_c[10], -4.4228e-8 * -0.31460 * 0.30190);
    for index in 0..25 {
        if index != 10 {
            assert_eq!(zero_c[index].to_bits(), free_c[index].to_bits());
        }
        assert_eq!(zero_s[index].to_bits(), free_s[index].to_bits());
    }
}

#[test]
fn mean_pole_historical_breakpoint_matches_public_reference() {
    for (epoch, expected_c21, expected_s21) in [
        (
            2455197.499999,
            -2.800576375308338e-12,
            -2.0341716099497407e-10,
        ),
        (2455197.5, -2.8005763475001803e-12, -2.03417160993e-10),
        (
            2455197.500001,
            -2.8005609901914863e-12,
            -2.0341582799102556e-10,
        ),
    ] {
        let (without_c, without_s) = coeffs(epoch, "tide_free", None);
        let (with_c, with_s) = coeffs(epoch, "tide_free", Some((0.1, 0.2)));
        close(with_c[11] - without_c[11], expected_c21);
        close(with_s[11] - without_s[11], expected_s21);
        for index in 0..25 {
            if index != 11 {
                assert_eq!(with_c[index].to_bits(), without_c[index].to_bits());
                assert_eq!(with_s[index].to_bits(), without_s[index].to_bits());
            }
        }
    }
}

#[test]
fn invalid_inputs_fail_without_a_force() {
    for (sun, moon, mu, radius, tt, system, pole) in [
        (SUN, MOON, 0.0, RADIUS, 2459669.5, "tide_free", None),
        (SUN, MOON, MU, -1.0, 2459669.5, "tide_free", None),
        ([0.0; 3], MOON, MU, RADIUS, 2459669.5, "tide_free", None),
        (
            SUN,
            [0.0, 0.0, 1000.0],
            MU,
            RADIUS,
            2459669.5,
            "tide_free",
            None,
        ),
        (SUN, MOON, MU, RADIUS, f64::NAN, "tide_free", None),
        (SUN, MOON, MU, RADIUS, 2459669.5, "mean_tide", None),
        (
            SUN,
            MOON,
            MU,
            RADIUS,
            2459669.5,
            "tide_free",
            Some((f64::NAN, 0.0)),
        ),
    ] {
        assert!(coefficients(
            sun,
            moon,
            mu,
            radius,
            tt,
            2459669.4992,
            system,
            pole,
            SUN_MU,
            MOON_MU
        )
        .is_err());
    }
    assert!(acceleration(
        [0.0; 3],
        SUN,
        MOON,
        MU,
        RADIUS,
        2459669.5,
        2459669.4992,
        "tide_free",
        None,
        SUN_MU,
        MOON_MU
    )
    .is_err());
}
