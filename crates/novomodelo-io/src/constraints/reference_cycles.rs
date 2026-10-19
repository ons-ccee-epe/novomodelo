//! Reference-cycle detection over a name graph. The caller names what the graph's entries are
//! (`"named-expression"` for the file's `expressions` array) and the noun leads the rejection
//! message.

use std::collections::HashMap;

/// Three-colour DFS marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Color {
    Unvisited,
    InProgress,
    Done,
}

/// Reject a reference cycle over `graph`, whose entries are the declared names
/// with the names each one references; a referenced name with no entry is ignored.
///
/// Iterative three-colour DFS: never recurses over the chain, so a deep chain
/// cannot overflow the stack. A self-reference is a cycle of length 1. The reported
/// cycle depends on the order of `graph` (the start order and each node's child
/// order), so a caller that needs a report independent of declaration order passes
/// the graph sorted by a declaration-independent key.
///
/// # Errors
///
/// Returns `Err("{noun} reference cycle detected: a -> b -> a")`, listing the full
/// path of the first cycle found.
pub(crate) fn detect_cycles(graph: &[(&str, Vec<&str>)], noun: &str) -> Result<(), String> {
    let adjacency: HashMap<&str, &[&str]> = graph
        .iter()
        .map(|(name, refs)| (*name, refs.as_slice()))
        .collect();

    let mut color: HashMap<&str, Color> = graph
        .iter()
        .map(|(name, _)| (*name, Color::Unvisited))
        .collect();

    let mut stack: Vec<(&str, usize)> = Vec::new();
    let mut path: Vec<&str> = Vec::new();

    for &(start, _) in graph {
        if color.get(start).copied() != Some(Color::Unvisited) {
            continue;
        }
        color.insert(start, Color::InProgress);
        path.push(start);
        stack.push((start, 0));

        while let Some((node, child_idx)) = stack.last().copied() {
            let neighbors: &[&str] = adjacency.get(node).copied().unwrap_or_default();
            if child_idx < neighbors.len() {
                if let Some(frame) = stack.last_mut() {
                    frame.1 += 1;
                }
                let child = neighbors[child_idx];
                match color.get(child).copied() {
                    Some(Color::InProgress) => return Err(cycle_message(noun, &path, child)),
                    Some(Color::Unvisited) => {
                        color.insert(child, Color::InProgress);
                        path.push(child);
                        stack.push((child, 0));
                    }
                    Some(Color::Done) | None => {}
                }
            } else {
                color.insert(node, Color::Done);
                path.pop();
                stack.pop();
            }
        }
    }

    Ok(())
}

/// Render the cycle path closing back to `back_to`, which is on the current path.
fn cycle_message(noun: &str, path: &[&str], back_to: &str) -> String {
    let pos = path.iter().position(|&n| n == back_to).unwrap_or(0);
    let mut cycle: Vec<&str> = path[pos..].to_vec();
    cycle.push(back_to);
    format!("{noun} reference cycle detected: {}", cycle.join(" -> "))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn detect_cycles_acyclic_chain_is_ok() {
        let graph = [("a", vec!["b"]), ("b", vec!["c"]), ("c", vec![])];
        assert!(detect_cycles(&graph, "named-expression").is_ok());
    }

    #[test]
    fn detect_cycles_diamond_is_ok() {
        let graph = [
            ("outer", vec!["a", "b"]),
            ("a", vec!["base"]),
            ("b", vec!["base"]),
            ("base", vec![]),
        ];
        assert!(detect_cycles(&graph, "named-expression").is_ok());
    }

    #[test]
    fn detect_cycles_two_node_cycle_names_both() {
        let graph = [("a", vec!["b"]), ("b", vec!["a"])];
        let err = detect_cycles(&graph, "named-expression").unwrap_err();
        assert!(err.contains('a'), "message: {err}");
        assert!(err.contains('b'), "message: {err}");
        assert!(err.contains("a -> b -> a"), "message: {err}");
    }

    #[test]
    fn detect_cycles_self_reference_is_cycle_of_one() {
        let graph = [("e", vec!["e"])];
        let err = detect_cycles(&graph, "named-expression").unwrap_err();
        assert!(err.contains('e'), "message: {err}");
        assert!(err.contains("e -> e"), "message: {err}");
    }

    #[test]
    fn detect_cycles_names_the_cycle_with_the_caller_noun() {
        let graph = [("a", vec!["b"]), ("b", vec!["a"])];
        let err = detect_cycles(&graph, "expression-parameter").unwrap_err();
        assert_eq!(
            err,
            "expression-parameter reference cycle detected: a -> b -> a"
        );
    }
}
