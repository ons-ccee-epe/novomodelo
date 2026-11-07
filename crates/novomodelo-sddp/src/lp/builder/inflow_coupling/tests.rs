use super::InflowCouplingSink;
use crate::indexer::HydroSys;

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "inflow-coupling sink: hydro index out of range")]
fn record_rejects_a_hydro_past_the_sink() {
    InflowCouplingSink::new(2).record(HydroSys::new(2), 0, -1.0);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "inflow-coupling sink: entry is not a base row of this stage")]
fn finish_rejects_a_row_past_the_base_rows() {
    let mut sink = InflowCouplingSink::new(1);
    sink.record(HydroSys::new(0), 5, -1.0);
    let _ = sink.finish(5);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "inflow-coupling sink: non-finite coefficient")]
fn finish_rejects_a_nan_coefficient() {
    let mut sink = InflowCouplingSink::new(1);
    sink.record(HydroSys::new(0), 0, f64::NAN);
    let _ = sink.finish(1);
}

#[test]
fn finish_orders_hydros_canonically_and_rows_ascending_within_a_hydro() {
    let mut sink = InflowCouplingSink::new(3);
    sink.record(HydroSys::new(2), 4, -4.0);
    sink.record(HydroSys::new(0), 7, -7.0);
    sink.record(HydroSys::new(2), 1, -1.0);
    sink.record(HydroSys::new(0), 2, -2.0);
    sink.record(HydroSys::new(0), 2, -2.5);

    let coupling = sink.finish(8);
    let entries = |h: usize| coupling.hydro_entries(HydroSys::new(h)).collect::<Vec<_>>();

    assert_eq!(coupling.n_hydros(), 3);
    assert_eq!(entries(0), vec![(2, -2.0), (2, -2.5), (7, -7.0)]);
    assert_eq!(entries(1), Vec::<(usize, f64)>::new());
    assert_eq!(entries(2), vec![(1, -1.0), (4, -4.0)]);
}

#[test]
fn finish_orders_entries_canonically_whatever_the_push_order() {
    let entries = [
        (0, 7, -7.0),
        (2, 4, -4.0),
        (0, 2, -2.0),
        (1, 6, -6.0),
        (2, 1, -1.0),
        (1, 3, -3.0),
        (0, 5, -5.0),
    ];
    let finished = |order: [usize; 7]| {
        let mut sink = InflowCouplingSink::new(3);
        for i in order {
            let (hydro, row, coefficient) = entries[i];
            sink.record(HydroSys::new(hydro), row, coefficient);
        }
        sink.finish(8)
    };

    let in_order = finished([0, 1, 2, 3, 4, 5, 6]);
    let permuted = finished([4, 1, 6, 0, 3, 5, 2]);

    assert_eq!(in_order.hydro_starts, permuted.hydro_starts);
    assert_eq!(in_order.rows, permuted.rows);
    let bits = |coefficients: &[f64]| coefficients.iter().map(|c| c.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&in_order.coefficients), bits(&permuted.coefficients));
}
