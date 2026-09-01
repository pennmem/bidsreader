# bidsreader

A Python library for reading and working with neuroimaging data stored in the [BIDS (Brain Imaging Data Structure)](https://bids.neuroimaging.io/) format. Provides a structured, object-oriented interface for loading EEG and iEEG data, events, electrodes, and channel metadata, with built-in support for MNE-Python and PTSA.

## Features

- Load BIDS-compliant EEG/iEEG datasets with minimal boilerplate
- Automatic detection of device type (EEG vs iEEG) and coordinate space
- Load events, electrodes, channels, raw data, and epochs through a unified reader API
- Filter trials by type across events DataFrames, MNE Raw, and MNE Epochs
- Convert between MNE and PTSA data formats
- Detect and convert EEG signal units (V, mV, uV, nV, etc.)
- Custom exception hierarchy for clear, actionable error messages

## Installation

### Prerequisites

- Python 3.10+
- Access to a BIDS-formatted dataset

### Install from source

```bash
git clone <repository-url>
cd bidsreader
pip install -e .
```

> **Note:** The project currently has no `pyproject.toml` or `setup.py`. To use it without one, add the project root to your Python path or install in development mode after creating a minimal `pyproject.toml` (see [Development Setup](#development-setup)).

### Dependencies

**Required:**

| Package    | Purpose                          |
|------------|----------------------------------|
| mne        | EEG data structures and I/O      |
| mne-bids   | BIDS path resolution and reading |
| pandas     | Tabular data (events, channels)  |
| numpy      | Numeric operations               |

**Optional:**

| Package | Purpose                                  |
|---------|------------------------------------------|
| ptsa    | PTSA TimeSeries conversion (`convert_unit`, `mne_*_to_ptsa`) |
| pytest  | Running the test suite                   |

Install all dependencies:

```bash
pip install mne mne-bids pandas numpy
# Optional
pip install ptsa pytest
```

## Quick Start

*See a more robust tutorial in tutorials/*

### Basic usage with CMLBIDSReader

```python
from bidsreader import CMLBIDSReader

# Initialize a reader (defaults to /data/LTP_BIDS for CML data)
reader = CMLBIDSReader(subject="R1001P", task="FR1", session=0)

# Load behavioral events
events = reader.load_events("beh")

# Load electrode locations
electrodes = reader.load_electrodes()

# Load channel metadata (intracranial requires acquisition type)
channels = reader.load_channels("monopolar")

# Load combined channel + electrode data
combined = reader.load_combined_channels("bipolar")

# Load raw EEG data (returns MNE Raw object)
raw = reader.load_raw(acquisition="monopolar")

# Load epochs around events
epochs = reader.load_epochs(tmin=-0.5, tmax=1.5, acquisition="monopolar")
```

### Using a custom BIDS root

```python
reader = CMLBIDSReader(
    root="/path/to/your/bids/dataset",
    subject="sub01",
    task="rest",
    session="01",
    device="eeg",
)
```

### Querying dataset metadata

```python
reader = CMLBIDSReader(root="/data/LTP_BIDS", subject="R1001P", task="FR1")

# List all subjects in the dataset
subjects = reader.get_dataset_subjects()

# List all tasks in the dataset
tasks = reader.get_dataset_tasks()

# List sessions for this subject
sessions = reader.get_subject_sessions()

# List tasks for this subject
subject_tasks = reader.get_subject_tasks()

# Get the highest session number across all subjects
max_session = reader.get_dataset_max_sessions(outlier_thresh=100)
```

### Changing reader fields after creation

```python
reader = CMLBIDSReader(subject="R1001P", task="FR1", session=0)

# Switch to a different session
reader.set_fields(session=1)

# Switch subject and task
reader.set_fields(subject="R1002P", task="catFR1")
```

### Filtering events by trial type

```python
from bidsreader import filter_events_df_by_trial_types, filter_by_trial_types

# Filter a DataFrame
events = reader.load_events("beh")
word_events, indices = filter_events_df_by_trial_types(events, ["WORD"])

# Filter across multiple data objects at once (with consistency checks)
filtered_df, filtered_raw_events, filtered_epochs, event_id, idx = filter_by_trial_types(
    ["WORD", "STIM"],
    events_df=events,
    epochs=epochs,
)
```

### Unit detection and conversion

```python
from bidsreader import detect_unit, get_scale_factor, convert_unit

# Detect the unit of an MNE object
unit = detect_unit(raw)  # e.g., "V"

# Get conversion factor
factor = get_scale_factor("V", "uV")  # 1_000_000.0

# Convert data to a target unit (returns a copy by default)
raw_uv = convert_unit(raw, "uV")
```

### Converting MNE data to PTSA TimeSeries

```python
from bidsreader import mne_epochs_to_ptsa, mne_raw_to_ptsa

# Convert epochs (requires events DataFrame with 'sample' column)
ts = mne_epochs_to_ptsa(epochs, events)

# Convert raw data (optionally select channels and time window)
ts = mne_raw_to_ptsa(raw, picks=["E1", "E2"], tmin=0.0, tmax=10.0)
```

## Architecture

### Package layout

The package is split in two. **Every reader lives in `readers/`** — that's the folder
to open first. `src/` holds the shared machinery the readers are built from.

```
bidsreader/
├── readers/     # one module per dataset. Start here.
└── src/         # shared machinery: exceptions, helpers, units, filtering, conversion
```

Both are re-exported from the top level, so day-to-day use is a single import:

```python
from bidsreader import CMLBIDSReader, NiaBIDSReader, detect_unit, FileNotFoundBIDSError
```

### Class hierarchy

```
BaseReader              # Base — BIDS path construction, metadata queries, field validation
├── CMLBIDSReader       # CML (Computational Memory Lab) datasets
└── NiaBIDSReader       # Nia Therapeutics BIDS conversions (written by niadatsci)
```

### Module overview

| Module           | Purpose                                                  |
|------------------|----------------------------------------------------------|
| `readers/basereader.py` | `BaseReader` class — shared BIDS logic and metadata queries |
| `readers/cmlbidsreader.py` | `CMLBIDSReader` — CML-specific loading and auto-detection |
| `readers/niabidsreader.py` | `NiaBIDSReader` — Nia conversions (BrainVision, physio, montage `acq-`) |
| `src/filtering.py`   | Trial-type filtering for DataFrames, MNE Raw, and Epochs |
| `src/convert.py`     | MNE to PTSA TimeSeries conversion                        |
| `src/units.py`       | Unit detection, scaling, and conversion                  |
| `src/helpers.py`     | Utility functions (validation, BIDS prefix handling, bipolar electrode merging) |
| `src/exc.py`         | Custom exception hierarchy                               |
| `src/_errorwrap.py`  | `@public_api` decorator for consistent exception wrapping |

### Exception hierarchy

All exceptions inherit from `BIDSReaderError`, so you can catch everything with a single handler:

```
BIDSReaderError
├── InvalidOptionError        # Invalid argument value
├── MissingRequiredFieldError # Required reader field not set
├── FileNotFoundBIDSError     # Expected BIDS file not found
├── AmbiguousMatchError       # Multiple files matched when one expected
├── DataParseError            # TSV/JSON parsing failure
├── DependencyError           # Optional dependency issue
└── ExternalLibraryError      # Unexpected error from MNE/pandas/etc.
```

```python
from bidsreader import BIDSReaderError, FileNotFoundBIDSError

try:
    events = reader.load_events()
except FileNotFoundBIDSError:
    print("Events file not found for this subject/session")
except BIDSReaderError as e:
    print(f"Something went wrong: {e}")
```

## Creating a New Reader

To support a different BIDS dataset, subclass `BaseReader` and implement your dataset-specific logic. Here is a step-by-step guide.

### Step 1: Create your reader class

Create a new file in the readers folder (e.g., `bidsreader/readers/myreader.py`). Note the
import paths: sibling readers are `.`, and the shared machinery in `src/` is `..src.`.

```python
import pandas as pd
import mne
from pathlib import Path
from typing import Optional, Union
from .basereader import BaseReader
from ..src._errorwrap import public_api
from ..src.helpers import validate_option
from ..src.exc import FileNotFoundBIDSError


class MyDatasetReader(BaseReader):
    """Reader for the My Dataset BIDS archive."""

    # Valid options for constrained fields
    VALID_DEVICES = ("eeg", "meg")

    def __init__(
        self,
        root: Optional[Union[str, Path]] = "/data/my_dataset",
        subject: Optional[str] = None,
        task: Optional[str] = None,
        session: Optional[str | int] = None,
        space: Optional[str] = None,
        acquisition: Optional[str] = None,
        device: Optional[str] = None,
    ):
        # Validate device before passing to base
        device = validate_option("device", device, self.VALID_DEVICES)
        super().__init__(
            root=root,
            subject=subject,
            task=task,
            session=session,
            space=space,
            acquisition=acquisition,
            device=device,
        )

    # --- Override auto-detection hooks ---

    def _determine_device(self) -> Optional[str]:
        """Infer device type from subject ID or dataset structure.

        Return None if it cannot be determined.
        """
        if self.subject is None:
            return None
        # Example: subjects starting with "MEG" use MEG
        if self.subject.startswith("MEG"):
            return "meg"
        return "eeg"

    def _determine_space(self) -> Optional[str]:
        """Infer coordinate space from files on disk.

        Return None or raise FileNotFoundBIDSError / AmbiguousMatchError
        if it cannot be determined.
        """
        # Implement dataset-specific logic here
        return "MNI152NLin2009aSym"

    # --- Add your loading methods ---

    @public_api
    def load_events(self) -> pd.DataFrame:
        """Load behavioral events for the current subject/session/task."""
        self._require(("subject", "task", "session", "device"), context="load_events")

        bp = self._bp(datatype="beh", suffix="beh", extension=".tsv")
        matches = bp.match()
        if not matches:
            raise FileNotFoundBIDSError(f"No events file found for {bp}")

        return pd.read_csv(matches[0].fpath, sep="\t")

    @public_api
    def load_raw(self) -> mne.io.BaseRaw:
        """Load raw continuous data."""
        from mne_bids import read_raw_bids

        self._require(("subject", "task", "session", "device"), context="load_raw")
        bp = self._bp(datatype=self.device)
        return read_raw_bids(bp)
```

### Step 2: Key patterns to follow

1. **Validate constrained fields in `__init__`** using `validate_option()` before calling `super().__init__()`.

2. **Override `_determine_device()` and `_determine_space()`** to enable automatic detection. These are called lazily the first time `reader.device` or `reader.space` is accessed. Return `None` if detection fails — the base class will emit a warning.

3. **Use `self._require(fields, context=...)`** at the start of each loading method to ensure the necessary fields are set before attempting file I/O.

4. **Use `self._bp(**kwargs)`** to construct `BIDSPath` objects for file matching. This handles BIDS-standard path construction using the reader's current field values.

5. **Decorate all public methods with `@public_api`** so that external exceptions (FileNotFoundError, JSONDecodeError, etc.) are automatically mapped to the `BIDSReaderError` hierarchy.

6. **Use `self._add_bids_prefix(field, value)`** when you need to manually construct BIDS-prefixed path segments (e.g., `"sub-001"`, `"ses-0"`).

### Step 3: Export your reader

Export it from both `__init__.py` files — [readers/\_\_init\_\_.py](bidsreader/readers/__init__.py)
so it sits with the other readers, and [\_\_init\_\_.py](bidsreader/__init__.py) so callers get
it from the top level:

```python
# bidsreader/readers/__init__.py
from .myreader import MyDatasetReader

# bidsreader/__init__.py
from .readers.myreader import MyDatasetReader
```

### Step 4: Write tests

Follow the patterns in [tests/conftest.py](tests/conftest.py) for fixtures and [tests/test_cmlbidsreader.py](tests/test_cmlbidsreader.py) for test structure. Key patterns:

- Use `tmp_path` fixtures to create temporary BIDS directory structures
- Use skip decorators for integration tests that require real data on disk
- Test both the happy path and error cases (missing fields, invalid options, missing files)

```python
import pytest
from bidsreader import MyDatasetReader

@pytest.fixture
def my_reader(tmp_path):
    return MyDatasetReader(root=tmp_path, subject="EEG001", task="rest", session=1)

def test_device_detection(my_reader):
    assert my_reader.device == "eeg"

def test_missing_field_raises(tmp_path):
    reader = MyDatasetReader(root=tmp_path, subject="EEG001", task="rest")
    reader.session = None
    with pytest.raises(Exception):
        reader.load_events()
```

## Development Setup

### Running tests

```bash
# Run all tests
python -m pytest tests/

# Run a specific test file
python -m pytest tests/test_basereader.py -v

# Run with output
python -m pytest tests/ -v -s
```

Integration tests that depend on real data at `/data/LTP_BIDS/` are skipped automatically when that data is not available.

### Creating a pyproject.toml (recommended)

If you want proper `pip install -e .` support, create a `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "bidsreader"
version = "0.1.0"
description = "Data loader and file reader for the OpenBIDS format"
requires-python = ">=3.10"
dependencies = [
    "mne",
    "mne-bids",
    "pandas",
    "numpy",
]

[project.optional-dependencies]
ptsa = ["ptsa"]
dev = ["pytest"]
```

Then install with:

```bash
pip install -e ".[dev]"
```

## API Reference

### BaseReader

| Method | Description |
|--------|-------------|
| `set_fields(**kwargs)` | Set multiple reader fields at once (chainable) |
| `get_dataset_subjects()` | List all subjects in the dataset |
| `get_dataset_tasks()` | List all tasks in the dataset |
| `get_subject_sessions()` | List sessions for the current subject |
| `get_subject_tasks()` | List tasks for the current subject |
| `get_dataset_max_sessions(outlier_thresh=None)` | Get highest session number across all subjects |

### CMLBIDSReader

Inherits all `BaseReader` methods, plus:

| Method | Description |
|--------|-------------|
| `is_intracranial()` | Returns `True` if device is `"ieeg"` |
| `load_events(event_type="beh")` | Load events TSV (`"beh"` or device-type events) |
| `load_electrodes()` | Load electrode coordinates TSV |
| `load_channels(acquisition=None)` | Load channel metadata TSV (iEEG requires `"monopolar"` or `"bipolar"`) |
| `load_combined_channels(acquisition=None)` | Merge channel + electrode data into one DataFrame |
| `load_coordsystem_desc()` | Load coordinate system JSON metadata |
| `load_raw(acquisition=None)` | Load raw continuous data (returns `mne.io.BaseRaw`) |
| `load_epochs(tmin, tmax, events=None, baseline=None, acquisition=None, event_repeated="merge", channels=None, preload=False)` | Create `mne.Epochs` from raw data and events |

### NiaBIDSReader

Reads the BIDS conversions written by [niadatsci](https://github.com/) (`niashare.bids_convert.NiaBIDSConverter`).
Inherits all `BaseReader` methods, plus:

| Method | Description |
|--------|-------------|
| `is_nia_dataset()` | Returns `True` if `root` holds a niadatsci-generated dataset |
| `first_sample()` | Device-clock sample at which the recording begins. Provenance — relates `sample` to the `eegoffset` columns; not needed for epoching |
| `units_status()` | `"calibrated"` if the signal is in µV, otherwise raw ADC counts |
| `list_available_physio()` | List the `recording-<label>` physio streams for this session |
| `load_events(coerce_numeric=True)` | Load events TSV (literal `n/a` preserved) |
| `load_events_json()` | Load the events sidecar (column descriptions, `Levels`) |
| `load_channels(coerce_numeric=True)` | Load channel metadata TSV |
| `load_sidecar()` | Load the `_eeg.json` sidecar, including `Nia*` provenance keys |
| `load_physio(recording=None)` | Load a continuous non-EEG timeseries (classifier or IMU) |
| `load_electrodes()` | Load electrode coordinates TSV (usually absent — see below) |
| `load_coordsystem_desc()` | Load coordinate system JSON (usually absent) |
| `load_participants()` / `load_scans()` / `load_dataset_description()` | Dataset- and session-level metadata |
| `load_raw(extension=None)` | Load raw BrainVision data (returns `mne.io.BaseRaw`) |
| `load_epochs(tmin, tmax, events=None, baseline=None, event_repeated="merge", channels=None, preload=False, extension=None)` | Create `mne.Epochs`, applying the `first_sample()` correction |
| `get_data_index(root=None, task=None)` | Session index with file paths and `Nia*` quality columns |

`root` must point at the **dataset directory** — `<bids_root>/<protocol>/<experiment>`, e.g.
`/bids/preclinical/ACL` — not at `<bids_root>`. Nia writes one dataset per experiment because
session numbers restart at zero per `(subject, experiment)`. `is_nia_dataset()` catches the
mistake cheaply:

```python
from bidsreader import NiaBIDSReader

reader = NiaBIDSReader(root="/bids/preclinical/ACL", subject="S002", session=0)
assert reader.is_nia_dataset()

events = reader.load_events()      # task is inferred from the dataset
epochs = reader.load_epochs(tmin=-0.5, tmax=1.0, events=events)
```

### Standalone Functions

| Function | Module | Description |
|----------|--------|-------------|
| `filter_events_df_by_trial_types(events_df, trial_types)` | `filtering` | Filter events DataFrame by trial type |
| `filter_raw_events_by_trial_types(raw, trial_types)` | `filtering` | Filter MNE Raw annotations by trial type |
| `filter_epochs_by_trial_types(epochs, trial_types)` | `filtering` | Filter MNE Epochs by trial type |
| `filter_by_trial_types(trial_types, *, events_df, raw, epochs)` | `filtering` | Filter multiple data objects with consistency checks |
| `detect_unit(data, current_unit=None)` | `units` | Detect or validate EEG data unit |
| `get_scale_factor(from_unit, to_unit)` | `units` | Get multiplicative conversion factor between units |
| `convert_unit(data, target, *, current_unit=None, copy=True)` | `units` | Convert EEG data to a target unit |
| `mne_epochs_to_ptsa(epochs, events)` | `convert` | Convert MNE Epochs to PTSA TimeSeries |
| `mne_raw_to_ptsa(raw, picks=None, tmin=None, tmax=None)` | `convert` | Convert MNE Raw to PTSA TimeSeries |

## Known Issues

### Nia event `sample` is an index into the recording

**Status:** resolved on the conversion side. Recorded here because the previous convention was
different, and a dataset converted before the change will now be **rejected** by the reader
rather than silently misread.

**The current contract.** `NiaBIDSConverter` writes event positions relative to the recording:

| Field | Meaning | Range |
|-------|---------|-------|
| `events.tsv` `sample` | Index into this recording, counting from zero at the first stored sample | `0 <= sample < raw.n_times`, or `n/a` |
| `events.tsv` `onset` | `sample / SamplingFrequency`, derived from `sample` so the two agree by construction | `>= 0`, or `n/a` |
| `eegoffset` / `eegoffset_cmd` | Absolute device-clock counter, free-running | unbounded |
| `NiaFirstSample` (`_eeg.json`) | Device sample at which the recording begins | provenance only |

So `sample` indexes an MNE `Raw` directly — `load_epochs()` applies no offset, and
`first_sample()` is not consulted when epoching. What `first_sample()` is for is relating the two
clocks: `eegoffset_cmd - first_sample() == sample`, which is how you match a BIDS event back to a
row in the source archive.

**Why this is called out.** BrainVision cannot store `first_samp`, so `read_raw_bids()` always
reports a recording starting at sample 0. Earlier conversions wrote `sample` as an **absolute
device index**, which meant using it to index the `Raw` produced a valid index pointing at the
wrong time — no exception, no warning, every epoch shifted by `NiaFirstSample`. The reader
compensated by subtracting it. Both halves of that are now gone.

**A stale tree fails loudly.** `dataset_description.json` carries no version, so the two
conventions cannot be told apart from metadata. Instead `load_epochs()` range-checks every
sample and raises `DataParseError` if any falls at or beyond `raw.n_times` — which is exactly
what old absolute-sample events look like. The error names the remedy: re-run the niadatsci
`share` step for that session. This mirrors an invariant the converter itself asserts
(`max(sample) < raw.n_times`), so the check cannot disagree with correctly converted output.

**Events with no position.** An event logged before the recording started — the command log can
begin before the EEG stream is up — gets `n/a` in both `sample` and `onset`. The row survives
intact, including its absolute `eegoffset_cmd`; only its position in this recording is unknown.

- `load_events()` emits a `RuntimeWarning`, because a column holding `n/a` stays strings and
  `df["sample"].max()` would silently compare strings rather than numbers.
- `load_epochs()` **raises** rather than dropping those rows, so a trial count cannot shrink
  unnoticed. The error gives the filter to apply:

```python
events = events[pd.to_numeric(events["sample"], errors="coerce").notna()]
```

### Related conversion-side quirks

Same category — decisions in the converter that the reader has to compensate for. Worth revisiting
together if the conversion is being changed:

| Quirk | Effect on the reader |
|-------|----------------------|
| The device's stim duration is renamed to `stim_duration`, because BIDS reserves `duration`. BIDS `duration` is then `n/a` unless the conversion ran with `--stim-duration-units`. | Callers reaching for `duration` get `n/a`. The reader documents this but cannot fix it. |
| `channels.tsv` `units` is the literal string `arbitrary` when the conversion ran without `--microvolts-per-lsb` — the signal is raw 15-bit ADC counts. | `bidsreader.src.units.detect_unit()` rejects `arbitrary`, so unit conversion is unavailable. Check `reader.units_status()` first. |
| `electrodes.tsv` / `coordsystem.json` are written only when the source `electrodes.csv` carries coordinates, which is usually not the case. | `load_electrodes()` / `load_coordsystem_desc()` raise `FileNotFoundBIDSError`. Treat both as optional. |

### Validation errors are reported as `ExternalLibraryError`

`@public_api` wraps bare `ValueError` into `ExternalLibraryError`
([_errorwrap.py](bidsreader/src/_errorwrap.py)), so some deliberate validation failures — an
unknown unit, an unknown field — surface as though a third-party library crashed. Exceptions
already in the hierarchy (`InvalidOptionError`, `MissingRequiredFieldError`) pass through
untouched. Catching `BIDSReaderError` is unaffected; catching `ValueError` will miss these.
`NiaBIDSReader` raises `DataParseError` for its own validation failures rather than relying on
the wrapper, so it differs from `CMLBIDSReader` here.

## License

TBD
