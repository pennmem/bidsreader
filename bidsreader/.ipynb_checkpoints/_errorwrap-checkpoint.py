# _errorwrap.py
from __future__ import annotations
from functools import wraps
import json
import pandas as pd

from .exc import (
    BIDSReaderError,
    InvalidOptionError,
    MissingRequiredFieldError,
    FileNotFoundBIDSError,
    AmbiguousMatchError,
    DataParseError,
    ExternalLibraryError,
)

def public_api(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)

        # If it's already one of yours, don't touch it.
        except BIDSReaderError:
            raise

        # Map common “expected” external exceptions to your hierarchy.
        except FileNotFoundError as e:
            raise FileNotFoundBIDSError(str(e)) from e

        except json.JSONDecodeError as e:
            raise DataParseError(f"Invalid JSON: {e}") from e

        except pd.errors.ParserError as e:
            raise DataParseError(f"Could not parse TSV/CSV: {e}") from e

        except KeyError as e:
            # Often means missing expected column like "trial_type"
            raise DataParseError(f"Missing expected key/column: {e}") from e

        except ValueError as e:
            # Be careful: ValueError is broad. Only map if you know it's "yours".
            # Otherwise wrap as ExternalLibraryError.
            raise ExternalLibraryError(str(e)) from e

        except Exception as e:
            # Last resort: you still guarantee your hierarchy.
            raise ExternalLibraryError(f"{type(e).__name__}: {e}") from e

    return wrapper