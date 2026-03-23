use chrono::{DateTime, NaiveDate, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

/// Result of temporal parsing across all 5 layers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalResult {
    /// Layer 1+2: A fully resolved calendar date.
    ExactDate(NaiveDate),

    /// Layer 3: A relative offset from the reference timestamp.
    RelativeDate {
        offset_days: i64,
        resolved: Option<NaiveDate>,
    },

    /// Layer 4: A fiscal/quarter date range.
    DateRange {
        start: NaiveDate,
        end: NaiveDate,
        label: String,
    },

    /// Layer 5: A duration expression.
    Duration {
        months: u32,
        days: u32,
        label: String,
    },
}

// ---------------------------------------------------------------------------
// Compiled regexes (built once)
// ---------------------------------------------------------------------------

static ISO_FULL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(\d{4})-(\d{2})-(\d{2})\b").unwrap());

static ISO_MONTH: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b(\d{4})-(\d{2})\b").unwrap());

// "March 11, 2026" or "Mar 11, 2026"
static WRITTEN_MDY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2}),?\s+(\d{4})\b").unwrap()
});

// "11 March 2026"
static WRITTEN_DMY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(\d{1,2})\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})\b").unwrap()
});

// "Mar 2026" / "March 2026"
static WRITTEN_MY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})\b").unwrap()
});

// Layer 3 — relative expressions
static REL_YESTERDAY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\byesterday\b").unwrap());

static REL_TODAY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\btoday\b").unwrap());

static REL_TOMORROW: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\btomorrow\b").unwrap());

static REL_N_DAYS_AGO: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(\d+)\s+days?\s+ago\b").unwrap());

static REL_N_WEEKS_AGO: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(\d+)\s+weeks?\s+ago\b").unwrap());

static REL_LAST_WEEK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\blast\s+week\b").unwrap());

static REL_LAST_MONTH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\blast\s+month\b").unwrap());

// Layer 4 — quarter / fiscal year
static QUARTER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bQ([1-4])\s+(\d{4})\b").unwrap());

static FISCAL_YEAR_LONG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bFY(\d{4})\b").unwrap());

static FISCAL_YEAR_SHORT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bFY(\d{2})\b").unwrap());

// Layer 5 — durations
static DUR_MONTHS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(?:for|over)\s+(\d+)\s+months?\b").unwrap());

static DUR_WEEKS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(?:for|over)\s+(\d+)\s+weeks?\b").unwrap());

static DUR_DAYS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(?:for|over)\s+(\d+)\s+days?\b").unwrap());

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse temporal expressions from `text`, resolving relative dates against
/// `reference`.  Returns all matches across all 5 layers.
pub fn parse_temporal(text: &str, reference: DateTime<Utc>) -> Vec<TemporalResult> {
    let ref_date = reference.date_naive();
    let mut results = Vec::new();

    layer1_iso(text, &mut results);
    layer2_written(text, &mut results);
    layer3_relative(text, ref_date, &mut results);
    layer4_fiscal(text, &mut results);
    layer5_duration(text, &mut results);

    results
}

// ---------------------------------------------------------------------------
// Layer 1 — ISO-8601
// ---------------------------------------------------------------------------

fn layer1_iso(text: &str, out: &mut Vec<TemporalResult>) {
    // Full dates first — we track their byte ranges so the month-only regex
    // doesn't duplicate them.
    let mut full_ranges: Vec<(usize, usize)> = Vec::new();

    for cap in ISO_FULL.captures_iter(text) {
        let m = cap.get(0).unwrap();
        full_ranges.push((m.start(), m.end()));

        let y: i32 = cap[1].parse().unwrap();
        let m_val: u32 = cap[2].parse().unwrap();
        let d: u32 = cap[3].parse().unwrap();
        if let Some(date) = NaiveDate::from_ymd_opt(y, m_val, d) {
            out.push(TemporalResult::ExactDate(date));
        }
    }

    for cap in ISO_MONTH.captures_iter(text) {
        let m = cap.get(0).unwrap();
        // Skip if this span is inside a full-date match.
        if full_ranges
            .iter()
            .any(|&(s, e)| m.start() >= s && m.end() <= e)
        {
            continue;
        }
        let y: i32 = cap[1].parse().unwrap();
        let m_val: u32 = cap[2].parse().unwrap();
        if let Some(date) = NaiveDate::from_ymd_opt(y, m_val, 1) {
            out.push(TemporalResult::ExactDate(date));
        }
    }
}

// ---------------------------------------------------------------------------
// Layer 2 — written dates
// ---------------------------------------------------------------------------

fn layer2_written(text: &str, out: &mut Vec<TemporalResult>) {
    // "March 11, 2026"
    for cap in WRITTEN_MDY.captures_iter(text) {
        let month = parse_month_name(&cap[1]);
        let day: u32 = cap[2].parse().unwrap();
        let year: i32 = cap[3].parse().unwrap();
        if let (Some(m), Some(date)) = (month, None::<NaiveDate>) {
            let _ = (m, date); // satisfy compiler
        }
        if let Some(m) = month
            && let Some(date) = NaiveDate::from_ymd_opt(year, m, day)
        {
            out.push(TemporalResult::ExactDate(date));
        }
    }

    // "11 March 2026"
    for cap in WRITTEN_DMY.captures_iter(text) {
        let day: u32 = cap[1].parse().unwrap();
        let month = parse_month_name(&cap[2]);
        let year: i32 = cap[3].parse().unwrap();
        if let Some(m) = month
            && let Some(date) = NaiveDate::from_ymd_opt(year, m, day)
        {
            out.push(TemporalResult::ExactDate(date));
        }
    }

    // "Mar 2026" — but skip if already matched by MDY (contains a day).
    // We use a simple heuristic: check that the match is not a substring of
    // a longer MDY match by verifying no digit precedes the month name in the
    // captured region.
    let mdy_ranges: Vec<(usize, usize)> = WRITTEN_MDY
        .find_iter(text)
        .map(|m| (m.start(), m.end()))
        .collect();
    let dmy_ranges: Vec<(usize, usize)> = WRITTEN_DMY
        .find_iter(text)
        .map(|m| (m.start(), m.end()))
        .collect();

    for cap in WRITTEN_MY.captures_iter(text) {
        let m = cap.get(0).unwrap();
        let overlaps_mdy = mdy_ranges
            .iter()
            .any(|&(s, e)| m.start() >= s && m.end() <= e);
        let overlaps_dmy = dmy_ranges
            .iter()
            .any(|&(s, e)| m.start() >= s && m.end() <= e);
        if overlaps_mdy || overlaps_dmy {
            continue;
        }
        let month = parse_month_name(&cap[1]);
        let year: i32 = cap[2].parse().unwrap();
        if let Some(mo) = month
            && let Some(date) = NaiveDate::from_ymd_opt(year, mo, 1)
        {
            out.push(TemporalResult::ExactDate(date));
        }
    }
}

fn parse_month_name(s: &str) -> Option<u32> {
    match s.to_ascii_lowercase().as_str() {
        "jan" | "january" => Some(1),
        "feb" | "february" => Some(2),
        "mar" | "march" => Some(3),
        "apr" | "april" => Some(4),
        "may" => Some(5),
        "jun" | "june" => Some(6),
        "jul" | "july" => Some(7),
        "aug" | "august" => Some(8),
        "sep" | "september" => Some(9),
        "oct" | "october" => Some(10),
        "nov" | "november" => Some(11),
        "dec" | "december" => Some(12),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Layer 3 — relative dates
// ---------------------------------------------------------------------------

fn layer3_relative(text: &str, ref_date: NaiveDate, out: &mut Vec<TemporalResult>) {
    if REL_TODAY.is_match(text) {
        out.push(TemporalResult::RelativeDate {
            offset_days: 0,
            resolved: Some(ref_date),
        });
    }

    if REL_YESTERDAY.is_match(text) {
        let d = ref_date - chrono::Duration::days(1);
        out.push(TemporalResult::RelativeDate {
            offset_days: -1,
            resolved: Some(d),
        });
    }

    if REL_TOMORROW.is_match(text) {
        let d = ref_date + chrono::Duration::days(1);
        out.push(TemporalResult::RelativeDate {
            offset_days: 1,
            resolved: Some(d),
        });
    }

    for cap in REL_N_DAYS_AGO.captures_iter(text) {
        let n: i64 = cap[1].parse().unwrap();
        let d = ref_date - chrono::Duration::days(n);
        out.push(TemporalResult::RelativeDate {
            offset_days: -n,
            resolved: Some(d),
        });
    }

    for cap in REL_N_WEEKS_AGO.captures_iter(text) {
        let n: i64 = cap[1].parse().unwrap();
        let days = n * 7;
        let d = ref_date - chrono::Duration::days(days);
        out.push(TemporalResult::RelativeDate {
            offset_days: -days,
            resolved: Some(d),
        });
    }

    if REL_LAST_WEEK.is_match(text) {
        let d = ref_date - chrono::Duration::days(7);
        out.push(TemporalResult::RelativeDate {
            offset_days: -7,
            resolved: Some(d),
        });
    }

    if REL_LAST_MONTH.is_match(text) {
        let d = ref_date - chrono::Duration::days(30);
        out.push(TemporalResult::RelativeDate {
            offset_days: -30,
            resolved: Some(d),
        });
    }
}

// ---------------------------------------------------------------------------
// Layer 4 — fiscal / quarter
// ---------------------------------------------------------------------------

fn layer4_fiscal(text: &str, out: &mut Vec<TemporalResult>) {
    for cap in QUARTER.captures_iter(text) {
        let q: u32 = cap[1].parse().unwrap();
        let year: i32 = cap[2].parse().unwrap();
        let (start_month, end_month) = match q {
            1 => (1, 3),
            2 => (4, 6),
            3 => (7, 9),
            4 => (10, 12),
            _ => continue,
        };
        if let (Some(start), Some(end)) = (
            NaiveDate::from_ymd_opt(year, start_month, 1),
            last_day_of_month(year, end_month),
        ) {
            out.push(TemporalResult::DateRange {
                start,
                end,
                label: format!("Q{q} {year}"),
            });
        }
    }

    for cap in FISCAL_YEAR_LONG.captures_iter(text) {
        let year: i32 = cap[1].parse().unwrap();
        if let (Some(start), Some(end)) = (
            NaiveDate::from_ymd_opt(year, 1, 1),
            NaiveDate::from_ymd_opt(year, 12, 31),
        ) {
            out.push(TemporalResult::DateRange {
                start,
                end,
                label: format!("FY{year}"),
            });
        }
    }

    // Short form FY26 -> 2026 (only if not already captured by long form).
    // We skip if the two-digit number also appeared as part of a 4-digit FY.
    let long_matches: Vec<String> = FISCAL_YEAR_LONG
        .captures_iter(text)
        .map(|c| c[1].to_string())
        .collect();

    for cap in FISCAL_YEAR_SHORT.captures_iter(text) {
        let short: &str = &cap[1];
        let year: i32 = 2000 + short.parse::<i32>().unwrap();
        let year_str = year.to_string();
        // Avoid double-counting FY2026 vs FY26
        if long_matches.contains(&year_str) {
            // Check if this specific match overlaps with a long-form match
            // by seeing if the full match text is 4 digits.
            continue;
        }
        if let (Some(start), Some(end)) = (
            NaiveDate::from_ymd_opt(year, 1, 1),
            NaiveDate::from_ymd_opt(year, 12, 31),
        ) {
            out.push(TemporalResult::DateRange {
                start,
                end,
                label: format!("FY{year}"),
            });
        }
    }
}

fn last_day_of_month(year: i32, month: u32) -> Option<NaiveDate> {
    if month == 12 {
        NaiveDate::from_ymd_opt(year, 12, 31)
    } else {
        NaiveDate::from_ymd_opt(year, month + 1, 1).map(|d| d - chrono::Duration::days(1))
    }
}

// ---------------------------------------------------------------------------
// Layer 5 — durations
// ---------------------------------------------------------------------------

fn layer5_duration(text: &str, out: &mut Vec<TemporalResult>) {
    for cap in DUR_MONTHS.captures_iter(text) {
        let n: u32 = cap[1].parse().unwrap();
        let full = cap.get(0).unwrap().as_str();
        out.push(TemporalResult::Duration {
            months: n,
            days: 0,
            label: full.to_string(),
        });
    }

    for cap in DUR_WEEKS.captures_iter(text) {
        let n: u32 = cap[1].parse().unwrap();
        let full = cap.get(0).unwrap().as_str();
        out.push(TemporalResult::Duration {
            months: 0,
            days: n * 7,
            label: full.to_string(),
        });
    }

    for cap in DUR_DAYS.captures_iter(text) {
        let n: u32 = cap[1].parse().unwrap();
        let full = cap.get(0).unwrap().as_str();
        out.push(TemporalResult::Duration {
            months: 0,
            days: n,
            label: full.to_string(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utc(y: i32, m: u32, d: u32) -> DateTime<Utc> {
        NaiveDate::from_ymd_opt(y, m, d)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
    }

    #[test]
    fn iso_full_date() {
        let r = parse_temporal("meeting on 2026-03-11", utc(2026, 1, 1));
        assert_eq!(r.len(), 1);
        assert_eq!(
            r[0],
            TemporalResult::ExactDate(NaiveDate::from_ymd_opt(2026, 3, 11).unwrap())
        );
    }

    #[test]
    fn iso_month_only() {
        let r = parse_temporal("report for 2026-03", utc(2026, 1, 1));
        assert_eq!(r.len(), 1);
        assert_eq!(
            r[0],
            TemporalResult::ExactDate(NaiveDate::from_ymd_opt(2026, 3, 1).unwrap())
        );
    }

    #[test]
    fn relative_yesterday() {
        let r = parse_temporal("as of yesterday", utc(2026, 3, 11));
        assert_eq!(r.len(), 1);
        assert_eq!(
            r[0],
            TemporalResult::RelativeDate {
                offset_days: -1,
                resolved: Some(NaiveDate::from_ymd_opt(2026, 3, 10).unwrap()),
            }
        );
    }
}
