use chrono::{DateTime, NaiveDate, Utc};
use ctxgraph_extract::temporal::{TemporalResult, parse_temporal};

fn utc(y: i32, m: u32, d: u32) -> DateTime<Utc> {
    NaiveDate::from_ymd_opt(y, m, d)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
}

fn nd(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

// ── Layer 1: ISO-8601 ──────────────────────────────────────────────────

#[test]
fn layer1_iso_full_date() {
    let r = parse_temporal("deadline is 2026-03-11", utc(2026, 1, 1));
    assert_eq!(r, vec![TemporalResult::ExactDate(nd(2026, 3, 11))]);
}

#[test]
fn layer1_iso_month_only() {
    let r = parse_temporal("planned for 2026-07", utc(2026, 1, 1));
    assert_eq!(r, vec![TemporalResult::ExactDate(nd(2026, 7, 1))]);
}

#[test]
fn layer1_iso_full_does_not_duplicate_month() {
    // "2026-03-11" should yield one ExactDate, not two (full + month).
    let r = parse_temporal("2026-03-11", utc(2026, 1, 1));
    assert_eq!(r.len(), 1);
}

#[test]
fn layer1_multiple_iso_dates() {
    let r = parse_temporal("from 2025-01-01 to 2025-12-31", utc(2026, 1, 1));
    assert_eq!(r.len(), 2);
    assert_eq!(r[0], TemporalResult::ExactDate(nd(2025, 1, 1)));
    assert_eq!(r[1], TemporalResult::ExactDate(nd(2025, 12, 31)));
}

// ── Layer 2: Written dates ─────────────────────────────────────────────

#[test]
fn layer2_month_day_year() {
    let r = parse_temporal("March 11, 2026 is the date", utc(2026, 1, 1));
    assert_eq!(r, vec![TemporalResult::ExactDate(nd(2026, 3, 11))]);
}

#[test]
fn layer2_abbreviated_month_day_year() {
    let r = parse_temporal("due by Jan 5, 2025", utc(2026, 1, 1));
    assert_eq!(r, vec![TemporalResult::ExactDate(nd(2025, 1, 5))]);
}

#[test]
fn layer2_day_month_year() {
    let r = parse_temporal("submitted 11 March 2026", utc(2026, 1, 1));
    assert_eq!(r, vec![TemporalResult::ExactDate(nd(2026, 3, 11))]);
}

#[test]
fn layer2_month_year_only() {
    let r = parse_temporal("report for Sep 2025", utc(2026, 1, 1));
    assert_eq!(r, vec![TemporalResult::ExactDate(nd(2025, 9, 1))]);
}

// ── Layer 3: Relative dates ────────────────────────────────────────────

#[test]
fn layer3_yesterday() {
    let r = parse_temporal("yesterday was busy", utc(2026, 3, 11));
    assert_eq!(
        r,
        vec![TemporalResult::RelativeDate {
            offset_days: -1,
            resolved: Some(nd(2026, 3, 10)),
        }]
    );
}

#[test]
fn layer3_n_days_ago() {
    let r = parse_temporal("3 days ago", utc(2026, 3, 11));
    assert_eq!(
        r,
        vec![TemporalResult::RelativeDate {
            offset_days: -3,
            resolved: Some(nd(2026, 3, 8)),
        }]
    );
}

#[test]
fn layer3_n_weeks_ago() {
    let r = parse_temporal("2 weeks ago", utc(2026, 3, 11));
    assert_eq!(
        r,
        vec![TemporalResult::RelativeDate {
            offset_days: -14,
            resolved: Some(nd(2026, 2, 25)),
        }]
    );
}

#[test]
fn layer3_last_week() {
    let r = parse_temporal("last week", utc(2026, 3, 11));
    assert_eq!(
        r,
        vec![TemporalResult::RelativeDate {
            offset_days: -7,
            resolved: Some(nd(2026, 3, 4)),
        }]
    );
}

#[test]
fn layer3_last_month() {
    let r = parse_temporal("last month", utc(2026, 3, 11));
    assert_eq!(
        r,
        vec![TemporalResult::RelativeDate {
            offset_days: -30,
            resolved: Some(nd(2026, 2, 9)),
        }]
    );
}

#[test]
fn layer3_today_and_tomorrow() {
    let r = parse_temporal("today and tomorrow", utc(2026, 3, 11));
    assert_eq!(r.len(), 2);
    assert_eq!(
        r[0],
        TemporalResult::RelativeDate {
            offset_days: 0,
            resolved: Some(nd(2026, 3, 11)),
        }
    );
    assert_eq!(
        r[1],
        TemporalResult::RelativeDate {
            offset_days: 1,
            resolved: Some(nd(2026, 3, 12)),
        }
    );
}

// ── Layer 4: Fiscal / quarter ──────────────────────────────────────────

#[test]
fn layer4_quarter() {
    let r = parse_temporal("revenue in Q1 2026", utc(2026, 1, 1));
    assert_eq!(
        r,
        vec![TemporalResult::DateRange {
            start: nd(2026, 1, 1),
            end: nd(2026, 3, 31),
            label: "Q1 2026".into(),
        }]
    );
}

#[test]
fn layer4_quarter_q3() {
    let r = parse_temporal("Q3 2025 results", utc(2026, 1, 1));
    assert_eq!(
        r,
        vec![TemporalResult::DateRange {
            start: nd(2025, 7, 1),
            end: nd(2025, 9, 30),
            label: "Q3 2025".into(),
        }]
    );
}

#[test]
fn layer4_fiscal_year_short() {
    let r = parse_temporal("FY26 budget", utc(2026, 1, 1));
    assert_eq!(
        r,
        vec![TemporalResult::DateRange {
            start: nd(2026, 1, 1),
            end: nd(2026, 12, 31),
            label: "FY2026".into(),
        }]
    );
}

#[test]
fn layer4_fiscal_year_long() {
    let r = parse_temporal("FY2026 plan", utc(2026, 1, 1));
    assert_eq!(
        r,
        vec![TemporalResult::DateRange {
            start: nd(2026, 1, 1),
            end: nd(2026, 12, 31),
            label: "FY2026".into(),
        }]
    );
}

// ── Layer 5: Durations ─────────────────────────────────────────────────

#[test]
fn layer5_for_months() {
    let r = parse_temporal("for 3 months", utc(2026, 1, 1));
    assert_eq!(
        r,
        vec![TemporalResult::Duration {
            months: 3,
            days: 0,
            label: "for 3 months".into(),
        }]
    );
}

#[test]
fn layer5_over_months() {
    let r = parse_temporal("over 6 months", utc(2026, 1, 1));
    assert_eq!(
        r,
        vec![TemporalResult::Duration {
            months: 6,
            days: 0,
            label: "over 6 months".into(),
        }]
    );
}

#[test]
fn layer5_for_weeks() {
    let r = parse_temporal("for 2 weeks", utc(2026, 1, 1));
    assert_eq!(
        r,
        vec![TemporalResult::Duration {
            months: 0,
            days: 14,
            label: "for 2 weeks".into(),
        }]
    );
}

#[test]
fn layer5_for_days() {
    let r = parse_temporal("for 10 days", utc(2026, 1, 1));
    assert_eq!(
        r,
        vec![TemporalResult::Duration {
            months: 0,
            days: 10,
            label: "for 10 days".into(),
        }]
    );
}

// ── Multi-layer ────────────────────────────────────────────────────────

#[test]
fn multi_layer_mixed() {
    let text = "From 2026-03-01 for 3 months, Q1 2026 results";
    let r = parse_temporal(text, utc(2026, 3, 11));
    // Should contain: ExactDate, DateRange (Q1), Duration
    assert!(r.iter().any(|t| matches!(t, TemporalResult::ExactDate(_))));
    assert!(
        r.iter()
            .any(|t| matches!(t, TemporalResult::DateRange { .. }))
    );
    assert!(
        r.iter()
            .any(|t| matches!(t, TemporalResult::Duration { .. }))
    );
}

#[test]
fn empty_input_returns_empty() {
    let r = parse_temporal("", utc(2026, 1, 1));
    assert!(r.is_empty());
}

#[test]
fn no_temporal_returns_empty() {
    let r = parse_temporal("the quick brown fox", utc(2026, 1, 1));
    assert!(r.is_empty());
}
