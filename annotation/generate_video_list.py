"""Generate a video catalogue Google Sheet from Drive folders."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from gspread.exceptions import WorksheetNotFound

from utils import VIDEO_EXTENSIONS, get_drive_service, get_sheet_client

COLUMN_NAMES = ["category", "method", "drive_id", "name", "mime_type"]
GOOGLE_FOLDER_MIME = "application/vnd.google-apps.folder"


def _is_video(name: str, mime_type: str) -> bool:
    name_l = (name or "").lower()
    mime_l = (mime_type or "").lower()
    return any(name_l.endswith(ext) for ext in VIDEO_EXTENSIONS) or mime_l.startswith("video/")


def iter_drive_videos(root_id: str) -> Iterable[Dict[str, str]]:
    service = get_drive_service()
    queue = [root_id]

    while queue:
        current = queue.pop()
        info = service.files().get(fileId=current, fields="name").execute()  # pylint: disable=no-member
        folder_name = info.get("name", current)
        print(f"  - scanning folder {folder_name} ({current})...")
        page_token = None
        while True:
            response = (
                service.files()  # pylint: disable=no-member
                .list(
                    q=f"'{current}' in parents and trashed=false",
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageSize=200,
                    pageToken=page_token,
                )
                .execute()
            )
            for item in response.get("files", []):
                mime = item.get("mimeType", "")
                if mime == GOOGLE_FOLDER_MIME:
                    queue.append(item["id"])
                    continue
                name = item.get("name", "")
                if _is_video(name, mime):
                    yield {
                        "drive_id": item["id"],
                        "name": name,
                        "mime_type": mime,
                    }
            page_token = response.get("nextPageToken")
            if not page_token:
                break


def _ensure_list(value):
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def build_catalog(mapping: Dict[str, Dict[str, object]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for category, folders in mapping.items():
        for method, folder_ids in folders.items():
            for folder_id in _ensure_list(folder_ids):
                info = get_drive_service().files().get(fileId=folder_id, fields="name").execute()  # pylint: disable=no-member
                folder_name = info.get("name", folder_id)
                print(f"Scanning {category}/{method} - {folder_name} ({folder_id})")
                count = 0
                for item in iter_drive_videos(folder_id):
                    rows.append(
                        {
                            "category": category,
                            "method": method,
                            "drive_id": item["drive_id"],
                            "name": item["name"],
                            "mime_type": item.get("mime_type", ""),
                        }
                    )
                    count += 1
                print(f"    -> found {count} video(s)")
    return rows


def write_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMN_NAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in COLUMN_NAMES})


def write_sheet(rows: List[Dict[str, str]], sheet_url: str, worksheet_name: str | None = None) -> None:
    client = get_sheet_client()
    sheet = client.open_by_url(sheet_url)
    if worksheet_name:
        try:
            worksheet = sheet.worksheet(worksheet_name)
        except WorksheetNotFound:
            worksheet = sheet.add_worksheet(title=worksheet_name, rows="1000", cols="20")
    else:
        worksheet = sheet.sheet1

    data = [COLUMN_NAMES] + [[row.get(col, "") for col in COLUMN_NAMES] for row in rows]
    worksheet.clear()
    worksheet.update(range_name="A1", values=data)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Drive video catalogue and write it to a Google Sheet.")
    parser.add_argument("--config", required=True, help="Path to mapping JSON file")
    parser.add_argument("--sheet-url", required=True, help="URL of the Google Sheet that will store the catalogue")
    parser.add_argument("--worksheet", help="Worksheet name (defaults to the first sheet)")
    parser.add_argument(
        "--output",
        help="Optional path to also write a local CSV copy for inspection",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1

    mapping = json.loads(config_path.read_text(encoding="utf-8"))
    rows = build_catalog(mapping)
    print(f"Discovered {len(rows)} video(s).")

    if args.output:
        write_csv(rows, Path(args.output))
        print(f"CSV written to {Path(args.output).resolve()}")

    write_sheet(rows, args.sheet_url, args.worksheet)
    target = args.worksheet or "sheet1"
    print(f"Sheet '{target}' updated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
