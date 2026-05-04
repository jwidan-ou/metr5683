from pathlib import Path
import datetime as dt
import re
import os
import numpy as np
import urllib.request

import s3fs
import pandas as pd

##############################################
# Title: NEXRAD Download Utilities
# Description: Functions for downloading and processing NEXRAD radar data (AI-assisted)
# Author: Jacob Widanski
# Date: 12 April 2026
##############################################

### GLOBAL CONFIG ###
NEXRAD_BUCKET = "unidata-nexrad-level2"
NEXRAD_HTTP_ROOT = f"https://{NEXRAD_BUCKET}.s3.amazonaws.com"
# Allow 4-character alphanumeric IDs (e.g., KTLX, NOP4, DAN1)
NEXRAD_KEY_RE = re.compile(r"([A-Z0-9]{4})(\d{8})_(\d{6})_V\d{2}")
# KCRI aliases
RADAR_SITE_ALIASES = {
    "KCRI": ["KCRI", "NOP3", "NOP4", "ROP3", "ROP4", "FOP1", "DAN1", "DOP1"],
}

def _expand_radar_sites(site):
    """
    Expand a radar site ID to include all known aliases.
    """
    site = str(site).upper()
    aliases = RADAR_SITE_ALIASES.get(site, [site])
    return list(dict.fromkeys(aliases))

def _parse_nexrad_key_time(key):
    """
    Extract the scan time from a NEXRAD S3 key.
    """
    name = Path(key).name
    match = NEXRAD_KEY_RE.search(name)
    if not match:
        return None
    return dt.datetime.strptime(f"{match.group(2)}{match.group(3)}", "%Y%m%d%H%M%S")

def _s3_to_https_url(s3_path):
    """
    Convert an S3 path to an HTTPS URL for direct download.
    """
    if s3_path.startswith("s3://"):
        path = s3_path[len("s3://"):]
    else:
        path = s3_path

    if path.startswith(f"{NEXRAD_BUCKET}/"):
        key = path[len(NEXRAD_BUCKET) + 1:]
    else:
        key = path
    return f"{NEXRAD_HTTP_ROOT}/{key}"

def _list_nexrad_keys_s3fs(aws, prefix):
    """
    List NEXRAD keys in S3 under the given prefix using s3fs.
    """
    s3_dir = f"{NEXRAD_BUCKET}/{prefix}"
    try:
        items = aws.ls(s3_dir, refresh=True)
    except FileNotFoundError:
        return []
    except Exception as exc:
        print(f"Failed listing {s3_dir}: {exc}")
        return []

    keys = []
    for item in items:
        item_str = str(item)
        if item_str.startswith("s3://"):
            item_str = item_str[len("s3://"):]
        if item_str.startswith(f"{NEXRAD_BUCKET}/"):
            item_str = item_str[len(NEXRAD_BUCKET) + 1:]
        keys.append(item_str)
    return keys

def _download_nexrad_file(url, out_file):
    """
    Download a NEXRAD file from the given URL to the specified output file path.
    """
    with urllib.request.urlopen(url) as response:
        with open(out_file, "wb") as out_f:
            out_f.write(response.read())

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calculate the distance in kilometers between two points
    specified in decimal degrees using the Haversine formula.
    """
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))

    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Apply formula
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    rad = 6371.0 # Earth radius in kilometers
    return rad * c

def build_time_windows_from_masked_counts(
    region_reports,
    filtered_counts=None,
    selected_dates=None,
    time_col="time",
):
    """
    Build time windows for NEXRAD downloads based on filtered hail reports and region reports.
    Parameters:
        filtered_counts: DataFrame with daily hail reports (indexed by date)
        region_reports: DataFrame with hail reports in the specified region (must include time_col)
        selected_dates: Optional list of specific dates to include (if None, use all dates in filtered_counts)
        time_col: Name of the column in region_reports that contains the report timestamps
    Returns:
        DataFrame with columns: date, start_utc, end_utc, n_reports
    """
    reports = region_reports.copy()
    reports[time_col] = pd.to_datetime(reports[time_col], errors="coerce")
    reports = reports.dropna(subset=[time_col])

    if selected_dates is None:
        dates = pd.to_datetime(filtered_counts.index).date
    else:
        dates = pd.to_datetime(selected_dates).date

    rows = []
    for day in dates:
        day_reports = reports.loc[reports[time_col].dt.date == day]
        if day_reports.empty:
            continue
        rows.append(
            {
                "date": pd.Timestamp(day).date(),
                "start_utc": day_reports[time_col].min().to_pydatetime(),
                "end_utc": day_reports[time_col].max().to_pydatetime(),
                "n_reports": int(len(day_reports)),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["date", "start_utc", "end_utc", "n_reports"]).sort_values("date").reset_index(drop=True)

    return pd.DataFrame(rows).sort_values("start_utc").reset_index(drop=True)

def download_nexrad_from_aws_windows(
    windows,
    radar_sites,
    out_dir,
    skip_existing=True,
    include_mdm=False,
    dry_run=False,
    verbose=True,
):
    if isinstance(radar_sites, str):
        radar_sites = [radar_sites]
    if windows.empty:
        return pd.DataFrame(
            columns=["date", "radar", "archive_site", "scan_time_utc", "key", "local_file", "status"]
        )

    requested_sites = [site.upper() for site in radar_sites]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    aws = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-east-1"})

    records = []
    seen_keys = set()

    for _, row in windows.iterrows():
        start_utc = pd.to_datetime(row["start_utc"]).to_pydatetime()
        end_utc = pd.to_datetime(row["end_utc"]).to_pydatetime()

        for requested_site in requested_sites:
            archive_sites = _expand_radar_sites(requested_site)

            for archive_site in archive_sites:
                prefix = f"{start_utc:%Y/%m/%d}/{archive_site}"
                keys = _list_nexrad_keys_s3fs(aws, prefix)

                for key in keys:
                    if key in seen_keys:
                        continue
                    name = Path(key).name

                    if (not include_mdm) and ("MDM" in name):
                        continue
                    scan_time = _parse_nexrad_key_time(key)
                    if scan_time is None or not (start_utc <= scan_time <= end_utc):
                        continue

                    local_file = out_path / name
                    exists = local_file.exists()

                    if exists and skip_existing:
                        status = "exists_skipped"
                    elif dry_run:
                        status = "would_download"
                    else:
                        url = _s3_to_https_url(f"{NEXRAD_BUCKET}/{key}")
                        _download_nexrad_file(url, local_file)
                        status = "downloaded"

                    seen_keys.add(key)
                    records.append(
                        {
                            "date": row["date"],
                            "radar": requested_site,
                            "archive_site": archive_site,
                            "scan_time_utc": scan_time,
                            "key": key,
                            "local_file": str(local_file),
                            "status": status,
                        }
                    )

                    if verbose:
                        print(f"[{requested_site}/{archive_site}] {scan_time} -> {status}: {name}")

    if not records:
        return pd.DataFrame(
            columns=["date", "radar", "archive_site", "scan_time_utc", "key", "local_file", "status"]
        )
    return pd.DataFrame(records).sort_values("scan_time_utc").reset_index(drop=True)

def download_nexrad_from_filtered_counts(
    region_reports,
    selected_dates,
    radar_sites,
    out_dir,
    filtered_counts=None,
    **download_kwargs,
):
    windows = build_time_windows_from_masked_counts(
        filtered_counts=filtered_counts,
        region_reports=region_reports,
        selected_dates=selected_dates,
    )
    manifest = download_nexrad_from_aws_windows(
        windows=windows,
        radar_sites=radar_sites,
        out_dir=out_dir,
        **download_kwargs,
    )
    return windows, manifest