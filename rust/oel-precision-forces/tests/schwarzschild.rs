use oel_precision_forces::perturbations::schwarzschild_acceleration;

const MU_KM3_S2: f64 = 398_600.4418;
const STATE: [f64; 6] = [7000.0, 200.0, 100.0, 0.1, 7.4, 0.8];

#[test]
fn matches_public_python_reference() {
    // sim.pro_perturbations.schwarzschild.schwarzschild_acceleration(STATE, MU_KM3_S2).
    // Fixed public numeric inputs; acceleration components are in km/s².
    let expected = [
        1.55779183236948178e-11,
        1.30840037263730153e-12,
        3.15742578423822126e-13,
    ];
    let actual = schwarzschild_acceleration(STATE, MU_KM3_S2).unwrap();
    for axis in 0..3 {
        // Retain the established native/reference force parity tolerance.
        assert!((actual[axis] - expected[axis]).abs() <= 2.0e-24);
    }
}

#[test]
fn rejects_invalid_state_mu_and_zero_radius() {
    let finite_error = "Schwarzschild requires a finite state and positive finite mu";
    for axis in 0..6 {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut state = STATE;
            state[axis] = value;
            assert_eq!(
                schwarzschild_acceleration(state, MU_KM3_S2).unwrap_err(),
                finite_error
            );
        }
    }
    for mu in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            schwarzschild_acceleration(STATE, mu).unwrap_err(),
            finite_error
        );
    }
    assert_eq!(
        schwarzschild_acceleration([0.0; 6], MU_KM3_S2).unwrap_err(),
        "Schwarzschild requires a positive geocentric radius"
    );
}
