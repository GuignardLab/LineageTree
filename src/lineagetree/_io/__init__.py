"""
I/O subpackage for LineageTree.

Contains functions for reading and writing lineage data in various formats.
"""

from . import _loaders
from . import _writers

# Optionally, re-export the key functions for easier import
from ._loaders import (
    read_from_ASTEC,
    read_from_binary,
    read_from_bmf,
    read_from_csv,
    read_from_mamut_xml,
    read_from_mastodon,
    read_from_mastodon_csv,
    read_from_tgmm_xml,
    read_from_txt_for_celegans,
    read_from_txt_for_celegans_BAO,
    read_from_txt_for_celegans_CAO,
)

__all__ = [
    "_loaders",
    "read_from_ASTEC",
    "read_from_binary",
    "read_from_bmf",
    "read_from_csv",
    "read_from_mamut_xml",
    "read_from_mastodon",
    "read_from_mastodon_csv",
    "read_from_tgmm_xml",
    "read_from_txt_for_celegans",
    "read_from_txt_for_celegans_BAO",
    "read_from_txt_for_celegans_CAO",
]
