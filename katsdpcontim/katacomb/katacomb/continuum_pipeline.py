from abc import ABCMeta, abstractmethod
from copy import deepcopy
import logging
import multiprocessing
from pretty import pretty

import six

import katdal
from katdal import DataSet
from katdal.datasources import DataSourceNotFound

from katacomb import (KatdalAdapter, obit_context, AIPSPath,
                      task_factory,
                      img_factory,
                      uv_factory,
                      uv_export,
                      uv_history_obs_description,
                      uv_history_selection,
                      export_calibration_solutions,
                      export_clean_components,
                      export_images)
from katacomb.aips_path import next_seq_nr
from katacomb.util import (fractional_bandwidth,
                           log_obit_err)
import katacomb.configuration as kc

log = logging.getLogger('katacomb')

# Standard MFImage output classes for UV and CLEAN images
UV_CLASS = "MFImag"
IMG_CLASS = "IClean"

# Dictionary of Pipeline Builders that `pipeline_factory`
# can use to create Pipelines. Use `register_builder` to populate
_pipeline_tags = {}


def pipeline_factory(tag, *args, **kwargs):
    """
    Parameters
    ----------
    tag : str
        Name of pipeline to instantiate.

    Returns
    -------
    pipeline : :class:`Pipeline`
        A pipeline implementation
    """

    # Try and find registered builders
    try:
        builder = _pipeline_tags[tag]
    except KeyError:
        # Nothing registered
        builder = tag
    else:
        # Found a builder class/function. Call it to instantiate something
        pipeline = builder(*args, **kwargs)

        # Did we get the right kind of thing?
        if not isinstance(pipeline, Pipeline):
            raise TypeError("'%s' did not return a valid Pipeline. "
                            "Got a %s instead."
                            % (builder, pipeline))

        return pipeline

    # I really don't know how to build the requested thing
    raise ValueError("I don't know how to build a '%s' pipeline" % builder)


def register_workmode(name):
    """
    Decorator function that registers a class or function
    under ``name`` for use by the pipeline_factory
    """
    def decorator(builder):
        if name in _pipeline_tags:
            raise ValueError("'%s' already registered as a "
                             "pipeline builder" % name)

        _pipeline_tags[name] = builder

        return builder

    return decorator


class Pipeline(object):
    """
    Defines an abstract Pipeline interface with single execute method
    that a user calls.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def execute(self):
        raise NotImplementedError


class PipelineImplementation(Pipeline):
    """
    This class encapsulates state and behaviour required for
    executing Pipelines
    """
    def __init__(self):
        """
        Initialise a Continuum Pipeline implementation

        Attributes
        ----------
        uvblavg_params : dict
            Dictionary of UV baseline averaging task parameters
            Defaults to :code:`{}`.
        mfimage_params : dict
            Dictionary of MFImage task parameters
            Defaults to :code:`{}`.
        nvispio : integer
            Number of AIPS visibilities per IO operation.
            Defaults to 10240.
        prtlv : integer
            Chattiness of Obit tasks (between 1=Quiet and  5=Verbose)
            Defaults to 1.
        disk : integer
            AIPS disk number to use
        odisk : integer
            FITS disk number to use for export
        """

        self.nvispio = 10240
        self.uvblavg_params = {}
        self.mfimage_params = {}
        self.prtlv = 1
        self.disk = 1
        self.odisk = 1

        self.cleanup_uv_files = []
        self.cleanup_img_files = []

    @abstractmethod
    def execute_implementation(self):
        raise NotImplementedError

    # Require implementers to provide a context manager
    @abstractmethod
    def __enter__(self):
        raise NotImplementedError

    # Require implementers to provide a context manager
    @abstractmethod
    def __exit__(self, evalue, etype, etraceback):
        raise NotImplementedError

    def execute(self):
        with obit_context(), self as ctx:
            ctx.execute_implementation()

    def _blavg_scan(self, scan_path):
        """
        Baseline average the AIPS UV File in ``scan_path``.

        Parameters
        ----------
        :class:`AIPSPath`
            Path of the AIPS UV file

        Returns
        -------
        :class:`AIPSPath`
            Path of the baseline averaged file
        """
        blavg_kwargs = scan_path.task_input_kwargs()
        blavg_path = scan_path.copy(aclass='uvav')
        blavg_kwargs.update(blavg_path.task_output_kwargs())
        blavg_kwargs['prtLv'] = self.prtlv

        blavg_kwargs.update(self.uvblavg_params)

        log.info("Time-dependent baseline averaging "
                 "'%s' to '%s'", scan_path, blavg_path)

        blavg = task_factory("UVBlAvg", **blavg_kwargs)
        with log_obit_err(log):
            blavg.go()

        return blavg_path

    def _run_mfimage(self, uv_path, uv_sources):
        """
        Run the MFImage task
        """

        with uv_factory(aips_path=uv_path, mode="r") as uvf:
            merge_desc = uvf.Desc.Dict

        # Run MFImage task on merged file,
        out_kwargs = uv_path.task_output_kwargs(name='',
                                                aclass=IMG_CLASS,
                                                seq=0)
        out2_kwargs = uv_path.task_output2_kwargs(name='',
                                                  aclass=UV_CLASS,
                                                  seq=0)

        mfimage_kwargs = {}
        # Setup input file
        mfimage_kwargs.update(uv_path.task_input_kwargs())
        # Output file 1 (clean file)
        mfimage_kwargs.update(out_kwargs)
        # Output file 2 (uv file)
        mfimage_kwargs.update(out2_kwargs)
        mfimage_kwargs.update({
            'maxFBW': fractional_bandwidth(merge_desc)/20.0,
            'nThreads': multiprocessing.cpu_count(),
            'prtLv': self.prtlv,
            'Sources': uv_sources
        })

        # Finally, override with default parameters
        mfimage_kwargs.update(self.mfimage_params)

        log.info("MFImage arguments %s" % pretty(mfimage_kwargs))

        mfimage = task_factory("MFImage", **mfimage_kwargs)
        # Send stdout from the task to the log
        with log_obit_err(log):
            mfimage.go()

    def _cleanup(self):
        """
        Remove any remaining UV, Clean, and Merged UVF files,
        if requested
        """

        # Clobber any result files requested
        for cleanup_file in set(self.cleanup_uv_files):
            with uv_factory(aips_path=cleanup_file, mode="w") as uvf:
                log.info("Zapping '%s'", uvf.aips_path)
                uvf.Zap()
        for cleanup_file in set(self.cleanup_img_files):
            with img_factory(aips_path=cleanup_file, mode="w") as imf:
                log.info("Zapping '%s'", imf.aips_path)
                imf.Zap()


class KatdalPipelineImplementation(PipelineImplementation):
    """
    This class has methods for executing Pipelines via a katdal
    interface to the data.

    These methods infer AIPSdisk file names for katdal targets
    and do the work of merging multiple target scans into a single
    merged and baseline-averaged scan for imaging.
    """

    def __init__(self, katdata):
        """
        Initialise a Continuum Pipeline with access to data
        using katdal.

        Parameters
        ----------
        katdata : :class:`katdal.Dataset`
            katdal Dataset object

        Attributes
        ----------
        katdal_select : dict
            Dictionary of katdal selection statements.
            Defaults to :code:`{}`.
        clobber : set or iterable
            Set or iterable of output files to clobber during
            conversion of katdal data to AIPS UV.
            Defaults to :code:`['scans', 'avgscans']`.
            Possible values include:
            1. `'scans'`, UV data files containing observational
                data for individual scans.
            2. `'avgscans'`, UV data files containing time-dependent
                baseline data for individual scans.
        """
        super(KatdalPipelineImplementation, self).__init__()
        self.ka = KatdalAdapter(katdata)
        self.katdal_select = {}
        self.clobber = ['scans', 'avgscans']
        self.merge_scans = False

    def _sanity_check_merge_blavg_descriptors(self, merge_uvf, blavg_uvf):
        """
        Sanity check the merge UV file's metadata
        against that of the blavg UV file. Mostly we check that the
        frequency information is the same.
        """
        merge_desc = merge_uvf.Desc.Dict
        merge_fq_kw = dict(merge_uvf.tables["AIPS FQ"].keywords)
        merge_fq_rows = merge_uvf.tables["AIPS FQ"].rows

        blavg_desc = blavg_uvf.Desc.Dict
        blavg_fq_kw = dict(blavg_uvf.tables["AIPS FQ"].keywords)
        blavg_fq_rows = blavg_uvf.tables["AIPS FQ"].rows

        # Compare
        # (1) UV FITS descriptor
        # (2) FQ table keywords
        # (3) FQ table rows

        descriptor_keys = ('inaxes', 'cdelt', 'crval', 'naxis', 'crota', 'crpix',
                           'ilocu', 'ilocv', 'ilocw', 'ilocb', 'iloct', 'ilocsu',
                           'jlocc', 'jlocs', 'jlocf', 'jlocif', 'jlocr', 'jlocd')

        diff = {k: (merge_desc[k], blavg_desc[k]) for k in descriptor_keys
                if not merge_desc[k] == blavg_desc[k]}

        if len(diff) > 0:
            raise ValueError("merge and averaged UV descriptors differ "
                             "on the following keys:\n%s" % pretty(diff))

        diff = {k: (merge_fq_kw[k], blavg_fq_kw[k])
                for k in blavg_fq_kw.keys()
                if not merge_fq_kw[k] == blavg_fq_kw[k]}

        if len(diff) > 0:
            raise ValueError("merge and averaged FQ table keywords differ "
                             "on the following keys:\n%s" % pretty(diff))

        if not len(merge_fq_rows) == len(blavg_fq_rows):
            raise ValueError("merge (%d) and averaged (%d) FQ "
                             "number of rows differ"
                             % (len(merge_fq_rows), len(blavg_fq_rows)))

        diff = [("row %d" % r, {k: (mr[k], br[k]) for k in br.keys()
                if not mr[k] == br[k]})
                for r, (mr, br)
                in enumerate(zip(merge_fq_rows, blavg_fq_rows))]
        diff = [(r, d) for r, d in diff if len(d) > 0]

        if len(diff) > 0:
            raise ValueError("merge and averaged FQ rows "
                             "differ as follows\n:%s" % pretty(diff))

    def _maybe_create_merge_uvf(self, merge_uvf, blavg_uvf, global_table_cmds):
        """
        Create the merge file if it hasn't yet been created,
        conditioning it with the baseline averaged file
        descriptor. Baseline averaged files
        have integration time as an additional random parameter
        so the merged file will need to take this into account.
        """

        # If we've performed channel averaging, the FREQ dimension
        # and FQ tables in the baseline averaged file will be smaller
        # than that of the scan file. This data must either be
        # (1) Used to condition a new merge UV file
        # (2) compared against the method in the existing merge UV file

        if merge_uvf is not None:
            return merge_uvf

        blavg_desc = blavg_uvf.Desc.Dict

        log.info("Creating '%s'",  self.uv_merge_path)

        # Use the FQ table rows and keywords to create
        # the merge UV file.
        blavg_table_cmds = deepcopy(global_table_cmds)

        blavg_fq_kw = dict(blavg_uvf.tables["AIPS FQ"].keywords)
        blavg_fq_rows = blavg_uvf.tables["AIPS FQ"].rows

        fq_cmd = blavg_table_cmds["AIPS FQ"]
        fq_cmd["keywords"] = blavg_fq_kw
        fq_cmd["rows"] = blavg_fq_rows

        # Create the UV object
        merge_uvf = uv_factory(aips_path=self.uv_merge_path,
                               mode="w",
                               nvispio=self.nvispio,
                               table_cmds=blavg_table_cmds,
                               desc=blavg_desc)

        # Write history
        uv_history_obs_description(self.ka, merge_uvf)
        uv_history_selection(self.katdal_select, merge_uvf)

        # Ensure merge and blavg file descriptors match on important points
        self._sanity_check_merge_blavg_descriptors(merge_uvf, blavg_uvf)

        return merge_uvf

    def _copy_scan_to_merge(self, merge_firstVis,
                            merge_uvf, blavg_uvf, nx_row):
        """
        Copy scan data to merged UV file
        """

        # Record the starting visibility
        # for this scan in the merge file

        blavg_desc = blavg_uvf.Desc.Dict
        blavg_nvis = blavg_desc['nvis']

        nx_row['START VIS'] = [merge_firstVis]

        for blavg_firstVis in six.moves.range(1, blavg_nvis+1, self.nvispio):
            # How many visibilities do we write in this iteration?
            numVisBuff = min(blavg_nvis+1 - blavg_firstVis, self.nvispio)

            # Update read file descriptor
            blavg_desc = blavg_uvf.Desc.Dict
            blavg_desc.update(numVisBuff=numVisBuff)
            blavg_uvf.Desc.Dict = blavg_desc

            # Update write file descriptor
            merge_desc = merge_uvf.Desc.Dict
            merge_desc.update(numVisBuff=numVisBuff)
            merge_uvf.Desc.Dict = merge_desc

            # Read, copy, write
            blavg_uvf.Read(firstVis=blavg_firstVis)
            merge_uvf.np_visbuf[:] = blavg_uvf.np_visbuf
            merge_uvf.Write(firstVis=merge_firstVis)

            # Update merge visibility
            merge_firstVis += numVisBuff

        # Record the ending visibility
        # for this scan in the merge file
        nx_row['END VIS'] = [merge_firstVis-1]

        # Append row to index table
        merge_uvf.tables["AIPS NX"].rows.append(nx_row)

        return merge_firstVis

    def _source_info(self):
        """
        Infer MFImage source names and file outputs.
        For each source, MFImage outputs UV data and a CLEAN file.
        """

        target_indices = self.ka.target_indices[:]

        # Use output_id for labels
        label = kc.get_config()['output_id']

        # Source names
        uv_sources = [s["SOURCE"][0].strip() for s in self.ka.uv_source_rows]

        uv_files = [AIPSPath(name=s, disk=self.disk, label=label,
                    aclass=UV_CLASS, atype="UV")
                    for s in uv_sources]

        clean_files = [AIPSPath(name=s, disk=self.disk, label=label,
                       aclass=IMG_CLASS, atype="MA")
                       for s in uv_sources]

        # Find a maximum sequence number referencing unassigned
        # catalogue numbers for all uv and clean files
        max_uv_seq_nr = max(next_seq_nr(f) for f in uv_files)
        max_clean_seq_nr = max(next_seq_nr(f) for f in clean_files)

        uv_files = [f.copy(seq=max_uv_seq_nr) for f in uv_files]
        clean_files = [f.copy(seq=max_clean_seq_nr) for f in clean_files]

        return uv_sources, target_indices, uv_files, clean_files

    def _select_and_infer_files(self):
        """
        Perform katdal selection and infer aips paths of:
        (1) imaging target names and indices
        (2) uv file for each target
        (3) clean file  for each target
        """

        self.katdal_select['reset'] = 'TFB'

        # Perform katdal selection
        self.ka.select(**self.katdal_select)

        # Fall over on empty selections
        if not self.ka.size > 0:
            raise ValueError("The katdal selection "
                             "produced an empty dataset"
                             "\n'%s'\n" % pretty(self.katdal_select))

        result_tuple = self._source_info()

        return result_tuple

    def _export_and_merge_scans(self):
        """
        1. Read scans from katdal
        2. Export scan data to an AIPS UV file
        3. Baseline average the file.
        4. Merge averaged AIPS UV file into a merge UV file.
        """

        # The merged UV observation file. We wait until
        # we have a baseline averaged file with which to condition it
        merge_uvf = None

        uv_mp = self.ka.aips_path(aclass='merge', name=kc.get_config()['cb_id'])
        self.uv_merge_path = uv_mp.copy(seq=next_seq_nr(uv_mp))

        global_desc = self.ka.uv_descriptor()
        global_table_cmds = self.ka.default_table_cmds()

        # FORTRAN indexing
        merge_firstVis = 1

        # Scan indices
        scan_indices = [int(si) for si in self.ka.scan_indices]

        # Export each scan individually, baseline averaging and merging it
        # into the final observation file.
        # NOTE: Loop over scan indices here rather than using the ka.scans
        # generator to avoid a conflict with the loop over ka.scans in uv_export.
        for si in scan_indices:
            # Select the current scan
            self.ka.select(scans=si)
            # Get path, with sequence based on scan index
            scan_path = self.uv_merge_path.copy(aclass='raw', seq=int(si))
            # Get the AIPS source for logging purposes
            aips_source = self.ka.catalogue[self.ka.target_indices[0]]
            aips_source_name = aips_source["SOURCE"][0].strip()

            log.info("Creating '%s'", scan_path)

            # Create a UV file for the scan and export to it
            with uv_factory(aips_path=scan_path, mode="w",
                            nvispio=self.nvispio,
                            table_cmds=global_table_cmds,
                            desc=global_desc) as uvf:

                uv_export(self.ka, uvf)

            # Retrieve the single scan index.
            # The time centroids and interval should be correct
            # but the visibility indices need to be repurposed
            scan_uvf = uv_factory(aips_path=scan_path, mode='r',
                                  nvispio=self.nvispio)

            assert len(scan_uvf.tables["AIPS NX"].rows) == 1
            nx_row = scan_uvf.tables["AIPS NX"].rows[0].copy()
            scan_desc = scan_uvf.Desc.Dict
            scan_nvis = scan_desc['nvis']

            # If we should be merging scans
            # just use the existing scan path and file
            if self.merge_scans:
                blavg_path = scan_path
                blavg_uvf = scan_uvf
            # Otherwise performing baseline averaging, deriving
            # a new scan path and file
            else:
                # Perform baseline averaging
                blavg_path = self._blavg_scan(scan_path)
                blavg_uvf = uv_factory(aips_path=blavg_path,
                                       mode='r',
                                       nvispio=self.nvispio)

            # Create the merge UV file, if necessary
            merge_uvf = self._maybe_create_merge_uvf(merge_uvf, blavg_uvf,
                                                     global_table_cmds)

            blavg_desc = blavg_uvf.Desc.Dict
            blavg_nvis = blavg_desc['nvis']

            # Record something about the baseline averaging process
            param_str = ', '.join("%s=%s" % (k, v)
                                  for k, v
                                  in self.uvblavg_params.items())

            blavg_history = ("Scan %d '%s' averaged "
                             "%s to %s visiblities. UVBlAvg(%s)" %
                             (si, aips_source_name, scan_nvis,
                              blavg_nvis, param_str))

            log.info(blavg_history)

            merge_uvf.append_history(blavg_history)

            log.info("Merging '%s' into '%s'", blavg_path, self.uv_merge_path)
            merge_firstVis = self._copy_scan_to_merge(merge_firstVis,
                                                      merge_uvf, blavg_uvf,
                                                      nx_row)

            # Remove scan once merged
            if 'scans' in self.clobber:
                log.info("Zapping '%s'", scan_uvf.aips_path)
                scan_uvf.Zap()
            else:
                scan_uvf.Close()

            # If merging scans for testing purposes, our
            # baseline averaged file will be the same as the
            # scan file, which was handled above, so don't
            # delete again. Otherwise default to
            # normal clobber handling.
            if not self.merge_scans:
                if 'avgscans' in self.clobber:
                    log.info("Zapping '%s'", blavg_uvf.aips_path)
                    blavg_uvf.Zap()
                else:
                    blavg_uvf.Close()

        # Write the index table
        merge_uvf.tables["AIPS NX"].write()

        # Create an empty calibration table
        merge_uvf.attach_CL_from_NX_table(self.ka.max_antenna_number)

        # Close merge file
        merge_uvf.close()


@register_workmode('continuum_export')
class KatdalExportPipeline(KatdalPipelineImplementation):

    def __init__(self, katdata, uvblavg_params={}, katdal_select={},
                 nvispio=10240, merge_scans=False):
        """
        Initialise a pipeline for UV export from katdal to AIPS UV
        TODO: Write a script to actually use this.
              At the moment it is just used in the unit tests.

        Parameters
        ----------
        katdata : :class:`katdal.Dataset`
            katdal Dataset object
        uvblavg_params : dict
            Dictionary of UV baseline averaging task parameters
        katdal_select : dict
            Dictionary of katdal selection statements.
        nvispio : integer
            Number of AIPS visibilities per IO operation.
        merge_scans : boolean
            Don't do BL dependant averaging if True.
        """

        super(KatdalExportPipeline, self).__init__(katdata)
        self.katdal_select = katdal_select
        self.uvblavg_params = uvblavg_params
        self.nvispio = nvispio
        self.merge_scans = merge_scans

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        if etype:
            log.exception('Exception executing continuum pipeline')
        self._cleanup()

    def execute_implementation(self):
        self._select_and_infer_files()
        self._export_and_merge_scans()


@register_workmode('online')
class OnlinePipeline(KatdalPipelineImplementation):

    def __init__(self, katdata, telstate, uvblavg_params={}, mfimage_params={},
                 katdal_select={}, nvispio=10240):
        """
        Initialise the Continuum Pipeline for MeerKAT system processing.

        Parameters
        ----------
        katdata : :class:`katdal.Dataset`
            katdal Dataset object
        telstate : :class:`katsdptelstate.TelescopeState`
            Telescope state or Telescope state view
        uvblavg_params : dict
            Dictionary of UV baseline averaging task parameters
        mfimage_params : dict
            Dictionary of MFImage task parameters
        katdal_select : dict
            Dictionary of katdal selection statements.
        nvispio : integer
            Number of AIPS visibilities per IO operation.
        """

        super(OnlinePipeline, self).__init__(katdata)
        self.telstate = telstate
        self.uvblavg_params = uvblavg_params
        self.mfimage_params = mfimage_params
        self.katdal_select = katdal_select
        self.nvispio = nvispio

        # Use highest numbered FITS disk for FITS output.
        self.odisk = len(kc.get_config()['fitsdirs'])
        self.prtlv = 1
        self.disk = 1

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        if etype:
            log.exception('Exception executing continuum pipeline')
        self._cleanup()

    def execute_implementation(self):
        result_tuple = self._select_and_infer_files()
        uv_sources, target_indices, uv_files, clean_files = result_tuple

        self._export_and_merge_scans()
        self.cleanup_uv_files.append(self.uv_merge_path)

        self._run_mfimage(self.uv_merge_path, uv_sources)
        self.cleanup_uv_files += uv_files
        self.cleanup_img_files += clean_files

        export_calibration_solutions(uv_files, self.ka,
                                     self.mfimage_params, self.telstate)
        export_clean_components(clean_files, target_indices,
                                self.ka, self.telstate)
        export_images(clean_files, target_indices,
                      self.odisk, self.ka)


@register_workmode('offline')
def build_offline_pipeline(data, **kwargs):
    """
    Decide based on the type of data how to
    build an offline pipeline instance and do it.

    Parameters
    ----------
    data : str or :class:`katdal.Dataset`
        file location or katdal Dataset object
        TODO: Allow for different types of data (eg UVFITS)
              so that this function actually does something.

    Returns
    -------
    :class:`Pipeline`
        A pipeline instance
    """
    if isinstance(data, DataSet):
        return KatdalOfflinePipeline(data, **kwargs)
    if isinstance(data, str):
        try:
            ds = katdal.open(data)
        except (IOError, DataSourceNotFound):
            pass
        else:
            return KatdalOfflinePipeline(ds, **kwargs)
    raise ValueError('Data type of %s not recognised for %s' % (type(data), data))


class KatdalOfflinePipeline(KatdalPipelineImplementation):
    def __init__(self, katdata, uvblavg_params={}, mfimage_params={},
                 katdal_select={}, nvispio=10240, prtlv=2,
                 clobber=set(['scans', 'avgscans']), reuse=False):
        """
        Initialise the Continuum Pipeline for offline imaging
        using a katdal dataset.

        Parameters
        ----------
        katdata : :class:`katdal.Dataset`
            katdal Dataset object
        uvblavg_params : dict
            Dictionary of UV baseline averaging task parameters
        mfimage_params : dict
            Dictionary of MFImage task parameters
        katdal_select : dict
            Dictionary of katdal selection statements.
        nvispio : integer
            Number of AIPS visibilities per IO operation.
        prtlv : integer
            Chattiness of Obit tasks
        clobber : set or iterable
            Set or iterable of output files to clobber from the aipsdisk.
            Possible values include:
            1. `'scans'`, UV data files containing observational
                data for individual scans.
            2. `'avgscans'`, UV data files containing time-dependent
                baseline data for individual scans.
            3. `'merge'`, UV data file containing merged averaged scans.
            4. `'clean'`, Output images from MFImage.
            5. `'mfimage'`, Output UV data file from MFImage."
        reuse : bool
            Are we reusing a previous katdal export in the aipsdisk?
        """

        super(KatdalOfflinePipeline, self).__init__(katdata)
        self.uvblavg_params = uvblavg_params
        self.mfimage_params = mfimage_params
        self.katdal_select = katdal_select
        self.nvispio = nvispio
        self.prtlv = prtlv
        self.clobber = clobber
        self.reuse = reuse

        self.odisk = len(kc.get_config()['fitsdirs'])

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        if etype:
            log.exception('Exception executing continuum pipeline')
        self._cleanup()

    def execute_implementation(self):
        result_tuple = self._select_and_infer_files()
        uv_sources, target_indices, uv_files, clean_files = result_tuple
        if "mfimage" in self.clobber:
            self.cleanup_uv_files += uv_files
        if "clean" in self.clobber:
            self.cleanup_img_files += clean_files
        # Update MFImage source selection
        self.mfimage_params['Sources'] = uv_sources
        # Find the highest numbered merge file if we are reusing
        if self.reuse:
            uv_mp = self.ka.aips_path(aclass='merge', name=kc.get_config()['cb_id'])
            # Find the merge file with the highest seq #
            hiseq = next_seq_nr(uv_mp) - 1
            # hiseq will be zero if the aipsdisk has no 'merge' file
            if hiseq == 0:
                raise ValueError("AIPS disk at '%s' has no 'merge' file to reuse." %
                                 (kc.get_config()['aipsdirs'][self.disk - 1][-1]))
            else:
                # Get the AIPS entry of the UV data to reuse
                self.uv_merge_path = uv_mp.copy(seq=hiseq)
                log.info("Re-using UV data in '%s' from AIPS disk: '%s'" %
                         (self.uv_merge_path, kc.get_config()['aipsdirs'][self.disk - 1][-1]))
        else:
            self._export_and_merge_scans()
        if "merge" in self.clobber:
            self.cleanup_uv_files.append(self.uv_merge_path)
        self._run_mfimage(self.uv_merge_path, uv_sources)
        export_images(clean_files, target_indices,
                      self.odisk, self.ka)
