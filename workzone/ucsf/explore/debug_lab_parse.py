"""Diagnose lab parse failure: find a line matching pat 214688334614038 and print it."""
from __future__ import annotations

import os
import re
from glob import glob

PAT_ID = "214688334614038"
lab_dir = "/labs/hulab/UCSF/rdb_new/Filtered_Lab_New"
shards = sorted(glob(os.path.join(lab_dir, "2016_*.txt")))

# Find first shard with hits and dump first hit verbatim + cleaned
_BAD_NUM = re.compile(r"(?<=[^\d,])(\d{1,3}),(?=\d{3}(?!\d))")
def clean(line): return _BAD_NUM.sub(r"\1", line)

for s in shards:
    with open(s, "r", encoding="latin-1") as fh:
        header = fh.readline()
        header_cols = header.rstrip("\n").split(",")
        i_pat = header_cols.index("Patient_ID")
        for line in fh:
            if PAT_ID not in line:
                continue
            cleaned = clean(line)
            parts_raw = line.rstrip("\n").split(",")
            parts_cln = cleaned.rstrip("\n").split(",")
            print("shard:", s)
            print("header cols:", len(header_cols))
            print("raw parts:", len(parts_raw), "cleaned parts:", len(parts_cln))
            print("Patient_ID col idx:", i_pat)
            print("raw Patient_ID field [%r]" % parts_raw[i_pat])
            print("cleaned Patient_ID field [%r]" % parts_cln[i_pat])
            # date / time / value / name fields
            for col in ["Lab_Collection_Date", "Lab_Collection_Time",
                        "Lab_Value", "Lab_Common_Name", "Lab_Name", "LOINC_Code",
                        "Lab_Order_Date", "Lab_Order_Time"]:
                if col in header_cols:
                    i = header_cols.index(col)
                    if i < len(parts_cln):
                        print(f"  {col}: [{parts_cln[i]!r}]")
            print("--- raw line ---")
            print(line[:500])
            print("--- cleaned ---")
            print(cleaned[:500])
            raise SystemExit
print("no hit in first 60 2016 shards")
