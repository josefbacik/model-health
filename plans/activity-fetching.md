# Plan: add activity fetching to the direct fetcher

> **Status:** known bug, design only, not yet implemented. Discovered 2026-04-08 while working on the race retro.

## The bug

`model-health fetch` does not fetch activities at all, even with `--force`. The activities parquet on disk is a frozen snapshot from before the migration to the `garmin-connect` crate.

### Evidence
- `src/fetch.rs::fetch_all` fetches `daily_health`, `performance_metrics`, then calls `fetch_weight_and_bp`. There is no `activities` code path. `grep activities src/fetch.rs` → no matches.
- The `garmin-connect` crate (`../garmin-connect/src/lib.rs`) exposes only auth + generic HTTP helpers (`get`, `get_json`, `post_json`). No activity abstraction exists; activity fetching has to be built on top using the raw endpoint.
- Git history confirms no file in this repo has ever had "activit" in its name. Activities have never been fetched by code in this repo. The on-disk parquet was populated by the user's old `garmin-cli` tool that lived alongside the early model-health work.
- File timestamps confirm: every file in `~/Library/Application Support/garmin/activities/*.parquet` is dated 2026-03-27 — the day of the `df2f1d6 Migrate to garmin-connect crate, add direct fetcher and validation` commit. They haven't been touched since.

### Symptom that surfaced it
The user fixed event-type labels in Garmin Connect (tagging turkey trots, missing marathons, etc. as races), then ran `model-health fetch --from 2017-01-01 --force` to refresh. `model-health races` still showed the same 13 labeled races as before. `daily_health/` was rewritten by the sync (timestamps from 13:49–13:52 same day) but `activities/` was untouched.

## Fix shape

Add a `fetch_activities` function in `src/fetch.rs`, wired into `fetch_all` next to the existing daily-health / perf-metrics loop.

### Endpoint
Garmin Connect's activity list is paginated:
```
GET /activitylist-service/activities/search/activities?start={offset}&limit={page_size}
```
Returns a JSON array of activities. Empty array means we've exhausted the list. Page size 20-50 is typical.

### Per-activity processing
For each returned JSON object:
1. Extract typed columns matching the existing `activities` parquet schema (see "Schema match" below). Key columns: `activity_id`, `activity_name`, `activity_type` (from the nested `activityType.typeKey`), `start_time_local`, `start_time_gmt`, `duration_sec`, `distance_m`, calories/HR if present, `location_name`, etc.
2. Store the full original JSON payload as `raw_json` so downstream code (`races.rs` reads `eventType.typeKey` from here) keeps working unchanged.
3. Bucket the activity by its start week (`YYYY-Wnn`) for partition assignment.

### Storage layout
Match the existing layout exactly: weekly-partitioned files at `activities/YYYY-Wnn.parquet` based on activity start date. **Confirm by listing one of the existing files before writing the implementation** — don't assume.

### Upsert semantics
On write to a weekly partition:
1. Read the existing partition (if any).
2. Concat with new rows.
3. Dedup by `activity_id`, **keeping the last** so re-fetched activities (with edited event types) overwrite the old rows.
4. Sort by `start_time_local` and write back.

This is the same pattern `daily_health` and `performance_metrics` already use; reuse the helper if one exists or factor one out.

### Re-fetch policy
Three options considered:

1. **Always re-fetch every week in the requested range under `--force`**, otherwise paginate from the start until we hit a known `activity_id` and stop (incremental forward fetch).
   - Pro: simple, correct, picks up edits anywhere when forced.
   - Con: under `--force` it's API-heavy.
2. **Pure incremental: only fetch new activities since the latest known id.**
   - Pro: fastest.
   - Con: never picks up event-type edits to older activities — *exactly the bug we just hit*. Reject.
3. **Combo: incremental forward fetch for new activities + always re-fetch the last N weeks (default 12) to catch recent edits + full rewrite under `--force`.**
   - Pro: catches the common case (recent edits) without forcing a full re-pull.
   - Con: more code paths.

**Recommendation: option 3.** The 12-week refresh window matches how often a runner is likely to look back and re-tag old activities. `--force` remains the escape hatch for edits older than 12 weeks (which is what triggered this bug — the user re-tagged a race from 2018).

### Schema match
**Read one of the existing `activities/*.parquet` files before writing the fetch code** and mirror its column set exactly. Defining the schema from scratch risks dropping columns that other code (or future tools) will need. Mirroring is more conservative.

The current schema (from `model-health profile` output earlier in this project) includes:
```
activity_id            i64
profile_id             i32
activity_name          str
activity_type          str
start_time_local       datetime[μs]
start_time_gmt         datetime[μs]
duration_sec           f64
distance_m             f64
calories               i32          (currently 100% null)
avg_hr                 i32          (currently 100% null)
max_hr                 i32          (currently 100% null)
avg_speed              f64
max_speed              f64
elevation_gain         f64
elevation_loss         f64
avg_cadence            f64
avg_power              i32          (currently 100% null)
normalized_power       i32          (currently 100% null)
training_effect        f64
training_load          f64
start_lat              f64
start_lon              f64
end_lat                f64
end_lon                f64
ground_contact_time    f64
vertical_oscillation   f64
stride_length          f64
location_name          str
raw_json               str
```

**Side note (out of scope but tempting):** the `calories`, `avg_hr`, `max_hr`, `avg_power`, `normalized_power` columns are 100% null in the existing parquet — meaning the old `garmin-cli` writer wasn't extracting them either, even though they're almost certainly in `raw_json`. While we're rewriting the fetcher we could populate them from the raw JSON. Worth doing but mark it as a separate small task so it doesn't bloat the core fix.

## Open questions

1. **Page size.** Garmin's API will tell us — start with 20 and tune if it's too slow.
2. **Rate limiting.** The existing `fetch_all` sleeps 500ms between cache misses. Same pattern for activities is fine.
3. **Should activity fetching happen when only `daily_health` is requested, or always?** Default: always, since they're cheap (page through, not per-day). Add a `--skip-activities` flag if anyone ever needs to opt out.
4. **Error handling.** If a single activity's JSON fails to parse, skip it with a warning rather than failing the whole fetch. Pattern is already in `races.rs::load_races`.
5. **How to test.** The user has a real Garmin account and can run it end-to-end after we build it. Don't bother with a mock.

## Estimated scope

- ~200-300 lines of new fetch logic in `src/fetch.rs`
- A small helper for weekly partition file naming
- Wiring into `fetch_all`
- Possibly factoring an `upsert_partition` helper if one isn't already present in the existing code

## Connection to current work

This is a hard blocker for the race-retro analysis being useful. The whole approach depends on race labels in the activities parquet matching what's in Garmin Connect, and right now they don't because the parquet hasn't been refreshed since the migration. Until this is fixed, every label edit the user makes in Garmin is invisible to `model-health races`.

Priority: should be fixed before any further race-retro iteration. After this is done, the user re-runs `model-health fetch --force`, the new labels flow through, and we can finally see the contrast/correlation reports against a complete labeled set.
