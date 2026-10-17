//! Declaration-order permutation invariance of every committed deck's and two
//! in-code studies' stage-LP template facts: permuting the input entity
//! declaration order leaves every stage-LP template fact group
//! ([`template_fact_groups`]) byte-identical.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::fmt::Write;

use novomodelo_sddp::StudySetup;
use novomodelo_sddp::build_generic_constraint_echo_rows;
use novomodelo_sddp::test_support::decks::{SLOW_DECKS, committed_decks};
use novomodelo_sddp::test_support::template_fact_groups;
use sha2::{Digest, Sha256};

use common::build_setup_in_code;
use common::fresh_setup_with;
use common::fresh_system_and_setup_with;
use common::in_code_studies::{
    ChronologicalNoiseSpec, chronological_noise_study, mixed_lead_anticipated_study,
};
use common::permute::permute_case;

const SEED: u64 = 0x5EED_C0BE_5EED_C0BE;

#[test]
fn every_deck_template_is_invariant_to_declaration_order() {
    let slow_tests_enabled = cfg!(feature = "slow-tests");
    let mut mismatches: Vec<(String, String)> = Vec::new();

    for deck in committed_decks() {
        if !slow_tests_enabled && SLOW_DECKS.contains(&deck.key.as_str()) {
            continue;
        }

        let base_setup = fresh_setup_with(&deck.dir, |_| {});
        let permuted_dir = permute_case(&deck.dir, SEED);
        let permuted_setup = fresh_setup_with(permuted_dir.path(), |_| {});

        for group in mismatched_groups(&base_setup, &permuted_setup, &deck.key) {
            mismatches.push((deck.key.clone(), group));
        }
    }

    assert!(
        mismatches.is_empty(),
        "template facts differ under declaration-order permutation for (deck, group) pairs:\n{}",
        mismatches
            .iter()
            .map(|(deck, group)| format!("  {deck}\t{group}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

fn mismatched_groups(
    base_setup: &StudySetup,
    reversed_setup: &StudySetup,
    label: &str,
) -> Vec<String> {
    let base = template_fact_groups(base_setup);
    let reversed = template_fact_groups(reversed_setup);
    assert!(
        base.keys().eq(reversed.keys()),
        "{label}: reversed fact-group key set differs from base (base={:?}, reversed={:?})",
        base.keys().collect::<Vec<_>>(),
        reversed.keys().collect::<Vec<_>>(),
    );
    base.iter()
        .filter(|(group, base_bytes)| reversed.get(*group) != Some(*base_bytes))
        .map(|(group, _)| (*group).to_string())
        .collect()
}

#[test]
fn every_in_code_study_template_is_invariant_to_declaration_order() {
    let mut mismatches: Vec<(String, String)> = Vec::new();

    let mixed_lead_label = "in-code/mixed-lead-anticipated";
    let (system, config) = mixed_lead_anticipated_study(false);
    let base_setup = build_setup_in_code(system, &config);
    let (system, config) = mixed_lead_anticipated_study(true);
    let reversed_setup = build_setup_in_code(system, &config);
    for group in mismatched_groups(&base_setup, &reversed_setup, mixed_lead_label) {
        mismatches.push((mixed_lead_label.to_string(), group));
    }

    let chronological_pumping_label = "in-code/chronological-pumping";
    let (system, config) = chronological_noise_study(&ChronologicalNoiseSpec {
        pumping_station: true,
        ..ChronologicalNoiseSpec::default()
    });
    let base_setup = build_setup_in_code(system, &config);
    let (system, config) = chronological_noise_study(&ChronologicalNoiseSpec {
        pumping_station: true,
        reverse_declaration_order: true,
        ..ChronologicalNoiseSpec::default()
    });
    let reversed_setup = build_setup_in_code(system, &config);
    for group in mismatched_groups(&base_setup, &reversed_setup, chronological_pumping_label) {
        mismatches.push((chronological_pumping_label.to_string(), group));
    }

    assert!(
        mismatches.is_empty(),
        "template facts differ under declaration-order reversal for (study, group) pairs:\n{}",
        mismatches
            .iter()
            .map(|(study, group)| format!("  {study}\t{group}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
#[ignore = "neutrality capture: run explicitly with --ignored"]
fn committed_deck_template_digests() {
    let decks = committed_decks();
    assert!(
        !decks.is_empty(),
        "committed_decks() must find at least one deck"
    );
    for deck in decks {
        let (system, setup) = fresh_system_and_setup_with(&deck.dir, |_| {});
        for (group, bytes) in template_fact_groups(&setup) {
            println!("DIGEST\t{}\t{group}\t{}", deck.key, sha256_hex(&bytes));
        }
        let rows = build_generic_constraint_echo_rows(&setup, &system);
        println!(
            "DIGEST\t{}\tgeneric_constraint_echo\t{}",
            deck.key,
            sha256_hex(format!("{rows:?}").as_bytes())
        );
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}
