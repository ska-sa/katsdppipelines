import collections

from katsdpcontim import UVFacade, AIPSPath, uv_factory


def uv_merge(katdata, uv_merge_file, uv_scan_files):
    """
    Merges a list of scan files into a single observation.

    katdata: :class:`KatdalAdapter`
        Katdal Adapter, used to condition the merge file
    uv_merge_file: :class:`AIPSPath`
        Merge file
    uv_scan_files: :class:`AIPSPath` or some sequence/iterable.
        Single scan file or sequence of files to merge.
    """

    if isinstance(uv_scan_files, collections.Sequence):
        pass
    elif isinstance(uv_scan_files, AIPSPath):
        uv_scan_files = [uv_scan_files]
    else:
        raise TypeError("uv_scan_files is '%s', but should be "
                        "an AIPSPath or an iterable of AIPSPaths." %
                        type(uv_scan_files))

    with open_uv(merge_file, mode='w') as merge:
        for scan_file in uv_scan_files:
            with open_uv(scan_file, mode='r') as scan:
                pass
