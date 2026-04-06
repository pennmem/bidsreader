# imports
import pandas as pd
from pathlib import Path
import re

def get_data_index(root, task):
    root = Path(root)
    rows = []

    pattern = re.compile(r"sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)_task-" + re.escape(task), re.IGNORECASE)

    for f in root.rglob(f"*task-{task}*beh.tsv"):
        m = pattern.search(f.name)
        if m:
            rows.append({"subject": m.group("sub"), "task": task, "session": m.group("ses")})

    return pd.DataFrame(rows).drop_duplicates().sort_values(["subject", "session"]).reset_index(drop=True)