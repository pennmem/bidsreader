class BIDSReaderError(Exception):
    """ Generic exception for all BIDS Reader exceptions """

class InvalidOptionError(BIDSReaderError, ValueError):
    """ Raised when an input is not among options. """

class MissingRequiredFieldError(BIDSReaderError, ValueError):
     """ Raised when a required field is missing when loading file using BIDSPath. """

class FileNotFoundBIDSError(BIDSReaderError, FileNotFoundError):
    """ Raised when a BIDS file is not found. """

class AmbiguousMatchError(BIDSReaderError, Exception):
    """ Raised when multiple files are returned when searching. """

class DataParseError(BIDSReaderError):
    """TSV/JSON parsing, schema issues, etc."""

class DependencyError(BIDSReaderError):
    """Errors originating from optional deps or incompatible versions."""

class ExternalLibraryError(BIDSReaderError):
    """Fallback wrapper when MNE/pandas/etc. throws something unexpected."""