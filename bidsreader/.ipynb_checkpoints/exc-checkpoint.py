class BIDSReaderError(Exception):
    """ Generic exception for all BIDS Reader exceptions """

class InvalidOptionError(BIDSReaderError, ValueError):
    pass

class MissingRequiredFieldError(BIDSReaderError, ValueError):
     """ Raised when a required field is missing when loading file using BIDSPath """

class FileResolutionError(BIDSReaderError):
    pass

class FileNotFoundBIDSError(FileResolutionError, FileNotFoundError):
    pass

class AmbiguousMatchError(FileResolutionError):
    pass

class DataParseError(BIDSReaderError):
    """TSV/JSON parsing, schema issues, etc."""

class DependencyError(BIDSReaderError):
    """Errors originating from optional deps or incompatible versions."""

class ExternalLibraryError(BIDSReaderError):
    """Fallback wrapper when MNE/pandas/etc. throws something unexpected."""