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
    let _ = sink.finish(&[0.0; 5], &[0.0; 5]);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "inflow-coupling sink: non-finite coefficient")]
fn finish_rejects_a_nan_coefficient() {
    let mut sink = InflowCouplingSink::new(1);
    sink.record(HydroSys::new(0), 0, f64::NAN);
    let _ = sink.finish(&[0.0; 1], &[0.0; 1]);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "inflow-coupling sink: duplicate row for one hydro")]
fn finish_rejects_a_duplicate_row_for_one_hydro() {
    let mut sink = InflowCouplingSink::new(1);
    sink.record(HydroSys::new(0), 2, -2.0);
    sink.record(HydroSys::new(0), 2, -2.5);
    let _ = sink.finish(&[0.0; 3], &[0.0; 3]);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "inflow-coupling sink: row bound slices differ in length")]
fn finish_rejects_row_bound_slices_of_different_lengths() {
    let _ = InflowCouplingSink::new(1).finish(&[0.0; 2], &[0.0; 1]);
}

#[test]
fn finish_orders_hydros_canonically_and_rows_ascending_within_a_hydro() {
    let mut sink = InflowCouplingSink::new(3);
    sink.record(HydroSys::new(2), 4, -4.0);
    sink.record(HydroSys::new(0), 7, -7.0);
    sink.record(HydroSys::new(2), 1, -1.0);
    sink.record(HydroSys::new(0), 2, -2.0);

    let coupling = sink.finish(&[0.0; 8], &[0.0; 8]);
    let entries = |h: usize| coupling.hydro_entries(HydroSys::new(h)).collect::<Vec<_>>();

    assert_eq!(coupling.n_hydros(), 3);
    assert_eq!(entries(0), vec![(2, -2.0), (7, -7.0)]);
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
        sink.finish(&[0.0; 8], &[0.0; 8])
    };

    let in_order = finished([0, 1, 2, 3, 4, 5, 6]);
    let permuted = finished([4, 1, 6, 0, 3, 5, 2]);

    assert_eq!(in_order.hydro_starts, permuted.hydro_starts);
    assert_eq!(in_order.positions, permuted.positions);
    let bits = |coefficients: &[f64]| coefficients.iter().map(|c| c.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&in_order.coefficients), bits(&permuted.coefficients));
}

#[test]
fn finish_lists_each_touched_row_once_with_its_unscaled_bounds() {
    let mut sink = InflowCouplingSink::new(3);
    sink.record(HydroSys::new(0), 4, -4.0);
    sink.record(HydroSys::new(2), 4, -4.5);
    sink.record(HydroSys::new(0), 1, -1.0);
    sink.record(HydroSys::new(1), 6, -6.0);
    let row_lower = [-0.5, 1.5, -2.5, 3.5, 4.25, -5.5, 6.75, 7.5];
    let row_upper = [0.5, 11.5, 2.5, -3.5, 14.25, 5.5, -16.75, 17.5];

    let coupling = sink.finish(&row_lower, &row_upper);

    assert_eq!(coupling.touched_rows(), [1, 4, 6]);
    let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(coupling.touched_row_lower()),
        bits(&[row_lower[1], row_lower[4], row_lower[6]])
    );
    assert_eq!(
        bits(coupling.touched_row_upper()),
        bits(&[row_upper[1], row_upper[4], row_upper[6]])
    );
}

#[test]
fn finish_places_each_entry_at_its_touched_row_position() {
    let mut sink = InflowCouplingSink::new(2);
    sink.record(HydroSys::new(0), 7, -1.0);
    sink.record(HydroSys::new(0), 2, -2.0);
    sink.record(HydroSys::new(1), 2, -3.0);
    sink.record(HydroSys::new(1), 9, -4.0);

    let coupling = sink.finish(&[0.0; 10], &[0.0; 10]);
    let shift_entries = |h: usize| {
        coupling
            .hydro_shift_entries(HydroSys::new(h))
            .collect::<Vec<_>>()
    };
    let entries = |h: usize| coupling.hydro_entries(HydroSys::new(h)).collect::<Vec<_>>();

    assert_eq!(coupling.touched_rows(), [2, 7, 9]);
    assert_eq!(shift_entries(0), vec![(0, -2.0), (1, -1.0)]);
    assert_eq!(shift_entries(1), vec![(0, -3.0), (2, -4.0)]);
    assert_eq!(entries(0), vec![(2, -2.0), (7, -1.0)]);
    assert_eq!(entries(1), vec![(2, -3.0), (9, -4.0)]);
}
