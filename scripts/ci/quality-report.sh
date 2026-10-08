#!/usr/bin/env bash
#
# quality-report.sh — code-quality HOTSPOT report (ADVISORY, read-only).
#
# The check-*.sh gates are a RATCHET: they prevent new drift and new
# un-justified debt, but they are binary pass/fail and do not rank the debt
# that already exists. This report fills that gap. It joins signals we already
# have into a single ranked "where to act first" table, so continuous-quality
# work is data-driven rather than hand-assembled.
#
# It installs nothing and runs nothing expensive — every signal is derived from
# git history, file size, and the same comment conventions the gates enforce.
#
# Signals, per production source file (crates/*/src/*.rs):
#   churn      — commits touching the file within CHURN_SINCE (default 12 months).
#                The dominant hotspot signal: a file changed often is read and
#                edited often, so its complexity is paid for repeatedly.
#   loc        — physical line count (a size / "surface area" proxy).
#   rationales — count of `// Rationale:` comments (= load-bearing shape/dead-code
#                #[allow] suppressions; a direct map of where complexity is
#                concentrated, produced by the E4 burndown).
#   fn_banners — indented box-drawing banner-comment lines BEFORE the first
#                #[cfg(test)] boundary: a cheap PROXY for the N5 "banner fencing
#                groups inside a function" signal. Top-level (column-0) dividers
#                are N5-OK and not counted. This is an over-approximation (it also
#                counts impl/mod-block dividers); the precise in-fn set is what
#                check-comment-banners.sh reports. Good enough as a 0.10-weight
#                hotspot proxy.
#
# Composite score (0..100): churn dominates, then size, then the two
# complexity-concentration proxies. Each raw signal is min-max normalized across
# the scanned files, then combined:
#   score = 100 * (0.50*churn_n + 0.20*loc_n + 0.20*rationale_n + 0.10*banner_n)
# The weights are a transparent default, not a law — tune them for the question
# at hand. Raw columns are printed so every score is auditable.
#
# Usage:
#   scripts/ci/quality-report.sh                # top 25, churn since 12 months ago
#   CHURN_SINCE="6 months ago" scripts/ci/quality-report.sh
#   TOP_N=0 scripts/ci/quality-report.sh        # TOP_N=0 prints every file
#
# Exit code: ALWAYS 0 (advisory — a report, never a gate).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPO_ROOT

readonly CHURN_SINCE="${CHURN_SINCE:-12 months ago}"
readonly TOP_N="${TOP_N:-25}"

# Production .rs source roots (mirrors the comment/doc gates' SCAN_DIRS).
readonly SCAN_DIRS=(
    "${REPO_ROOT}/crates/novomodelo-core/src"
    "${REPO_ROOT}/crates/novomodelo-io/src"
    "${REPO_ROOT}/crates/novomodelo-solver/src"
    "${REPO_ROOT}/crates/novomodelo-comm/src"
    "${REPO_ROOT}/crates/novomodelo-stochastic/src"
    "${REPO_ROOT}/crates/novomodelo-sddp/src"
    "${REPO_ROOT}/crates/novomodelo-cli/src"
    "${REPO_ROOT}/crates/novomodelo-python/src"
    "${REPO_ROOT}/crates/novomodelo-mcp/src"
    "${REPO_ROOT}/crates/novomodelo-tui/src"
)

# --- Churn map: one git pass over all source paths, counted per file. --------
# `--name-only --pretty=format:` lists every touched path per commit; sort|uniq
# counts them. Restricted to the source pathspec so non-source churn is ignored.
declare -A CHURN
while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    CHURN["$path"]=$(( ${CHURN["$path"]:-0} + 1 ))
done < <(git -C "$REPO_ROOT" log --since="$CHURN_SINCE" --name-only \
    --pretty=format: -- 'crates/*/src/*.rs' 2>/dev/null | sort)

# --- Per-file raw signals: emit "churn<TAB>loc<TAB>rationales<TAB>fn_banners<TAB>relpath". ---
emit_rows() {
    local dir file rel churn loc rats banners
    for dir in "${SCAN_DIRS[@]}"; do
        [[ -d "$dir" ]] || continue
        while IFS= read -r -d '' file; do
            rel="${file#"${REPO_ROOT}/"}"
            churn="${CHURN[$rel]:-0}"
            loc=$(wc -l < "$file")
            rats=$(grep -c '// Rationale:' "$file" 2>/dev/null || true)
            # Indented box-drawing banners before the first cfg(test) boundary.
            banners=$(awk '/^[[:space:]]*#\[cfg\((all\()?test[,)]/{exit}
                {print}' "$file" \
                | grep -cP '^[[:space:]]+//.*[\x{2500}-\x{257F}]' 2>/dev/null || true)
            printf '%s\t%s\t%s\t%s\t%s\n' "$churn" "$loc" "$rats" "$banners" "$rel"
        done < <(find "$dir" -name "*.rs" -print0)
    done
}

# --- Normalize, score, sort, print. ------------------------------------------
emit_rows | awk -F'\t' -v top="$TOP_N" -v since="$CHURN_SINCE" '
    {
        churn[NR]=$1; loc[NR]=$2; rat[NR]=$3; ban[NR]=$4; file[NR]=$5
        if ($1+0>mc) mc=$1+0; if ($2+0>ml) ml=$2+0
        if ($3+0>mr) mr=$3+0; if ($4+0>mb) mb=$4+0
    }
    END {
        n=NR
        for (i=1;i<=n;i++) {
            cn=(mc>0?churn[i]/mc:0); ln=(ml>0?loc[i]/ml:0)
            rn=(mr>0?rat[i]/mr:0);  bn=(mb>0?ban[i]/mb:0)
            score[i]=100*(0.50*cn+0.20*ln+0.20*rn+0.10*bn)
            idx[i]=i
        }
        # Descending insertion sort on score (n is small — hundreds of files).
        for (i=2;i<=n;i++){ k=idx[i]; j=i-1
            while (j>=1 && score[idx[j]]<score[k]){ idx[j+1]=idx[j]; j-- }
            idx[j+1]=k }
        printf "Code-quality hotspots (churn since %s). Higher score = act first.\n\n", since
        printf "%-5s %-6s %-7s %-5s %-5s %-7s  %s\n", "RANK","SCORE","churn","loc","rats","banners","FILE"
        printf "%-5s %-6s %-7s %-5s %-5s %-7s  %s\n", "----","-----","-----","---","----","-------","----"
        shown=0
        for (r=1;r<=n;r++){ i=idx[r]
            if (top+0>0 && shown>=top+0) break
            if (score[i]<0.5) continue   # skip files with no signal at all
            printf "%-5d %-6.1f %-7d %-5d %-5d %-7d  %s\n", r, score[i], churn[i], loc[i], rat[i], ban[i], file[i]
            shown++
        }
        printf "\nscore = 100*(0.50*churn + 0.20*loc + 0.20*rationales + 0.10*fn_banners), each min-max normalized.\n"
        printf "churn=commits touching the file; rats=// Rationale: suppressions; banners=indented box-drawing dividers (proxy for in-fn N5 candidates; exact set in check-comment-banners.sh).\n"
        printf "Advisory: this report never fails the build. Tune CHURN_SINCE / TOP_N / weights for the question at hand.\n"
    }
'

# E8 comment-bloat advisory — ranked candidates for the comment-skeptic pass.
echo ""
bash "${REPO_ROOT}/scripts/ci/check-comment-bloat.sh" "${REPO_ROOT}"

exit 0
