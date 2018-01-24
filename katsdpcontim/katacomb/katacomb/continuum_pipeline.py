from copy import deepcopy
import logging
from pretty import pretty, pprint

import six

log = logging.getLogger('katacomb')

import katacomb
from katacomb import (KatdalAdapter, obit_context, AIPSPath,
                        task_factory,
                        uv_factory,
                        uv_export,
                        uv_history_obs_description,
                        uv_history_selection,
                        export_calibration_solutions,
                        export_clean_components)
from katacomb.aips_path import next_seq_nr
from katacomb.util import (parse_python_assigns,
                        log_exception,
                        post_process_args,
                        fractional_bandwidth)

# Standard MFImage output classes for UV and CLEAN images
UV_CLASS = "MFImag"
IMG_CLASS = "IClean"

class ContinuumPipeline(object):
    def __init__(self, katdata, telstate, **kwargs):
        self.ka = KatdalAdapter(katdata)
        self.telstate = telstate

        self.nvispio = kwargs.pop("nvispio", 1024)
        self.katdal_select = kwargs.pop("katdal_select", {})
        self.uvblavg_params = kwargs.pop("uvblavg_params", {})
        self.mfimage_params = kwargs.pop("mfimage_params", {})
        self.clobber = kwargs.pop("clobber", set(['scans', 'avgscans']))

    def blavg_scan(self, scan_path):
        """
        Baseline
        """
        blavg_kwargs = scan_path.task_input_kwargs()
        blavg_path = scan_path.copy(aclass='uvav')
        blavg_kwargs.update(blavg_path.task_output_kwargs())

        blavg_kwargs.update(self.uvblavg_params)

        log.info("Time-dependent baseline averaging "
                "'%s' to '%s'", scan_path, blavg_path)

        blavg = task_factory("UVBlAvg", **blavg_kwargs)
        blavg.go()

        return blavg_path

    def export_scans(self):
        with obit_context():
            self._export_scans()

    def _export_scans(self):
        # The merged UV observation file. We wait until
        # we have a baseline averaged file to condition it with
        merge_uvf = None

        uv_mp = self.ka.aips_path(aclass='merge')
        self.uv_merge_path = uv_mp.copy(seq=next_seq_nr(uv_mp))

        # Perform katdal selection
        # retrieving selected scan indices as python ints
        # so that we can do per scan selection
        self.ka.select(**self.katdal_select)

        scan_indices = [int(si) for si in self.ka.scan_indices]

        # Fall over on empty selections
        if not self.ka.size > 0:
            raise ValueError("The katdal selection "
                            "produced an empty dataset"
                            "\n'%s'\n" % pretty(self.katdal_select))

        global_desc = self.ka.uv_descriptor()
        global_table_cmds = self.ka.default_table_cmds()

        # FORTRAN indexing
        merge_firstVis = 1

        # Export each scan individually,
        # baseline averaging and merging it
        # into the final observation file
        for si in scan_indices:
            # Clear katdal selection and set to global selection
            self.ka.select()
            self.ka.select(**self.katdal_select)

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

                self.ka.select(scans=si)
                uv_export(self.ka, uvf)

            # Perform baseline averaging
            blavg_path = self.blavg_scan(scan_path)

            # Retrieve the single scan index.
            # The time centroids and interval should be correct
            # but the visibility indices need to be repurposed
            scan_uvf = uv_factory(aips_path=scan_path, mode='r',
                                            nvispio=self.nvispio)

            assert len(scan_uvf.tables["AIPS NX"].rows) == 1
            nx_row = scan_uvf.tables["AIPS NX"].rows[0].copy()
            scan_desc = scan_uvf.Desc.Dict
            scan_nvis = scan_desc['nvis']

            # If we've performed channel averaging, the FREQ dimension
            # and FQ tables in the baseline averaged file will be smaller
            # than that of the scan file. This data must either be
            # (1) Used to condition a new merge UV file
            # (2) compared against the method in the existing merge UV file
            blavg_uvf = uv_factory(aips_path=blavg_path, mode='r',
                                              nvispio=self.nvispio)
            blavg_desc = blavg_uvf.Desc.Dict
            blavg_nvis = blavg_desc['nvis']

            # Record something about the baseline averaging process
            param_str = ', '.join("%s=%s" % (k,v)
                                 for k,v
                                 in self.uvblavg_params.items())

            blavg_history = ("Scan %d '%s' averaged "
                             "%s to %s visiblities. UVBlAvg(%s)" %
                                (si, aips_source_name,
                                 scan_nvis, blavg_nvis,
                                 param_str))

            log.info(blavg_history)

            blavg_fq_keywords = dict(blavg_uvf.tables["AIPS FQ"].keywords)
            blavg_fq_rows = blavg_uvf.tables["AIPS FQ"].rows

            # Get the merge file if it hasn't yet been created,
            # conditioning it with the baseline averaged file
            # descriptor. Baseline averaged files
            # have integration time as an additional random parameter
            # so the merged file will need to take this into account.
            if merge_uvf is None:
                log.info("Creating '%s'",  self.uv_merge_path)

                # Use the FQ table rows and keywords to create
                # the merge UV file.
                blavg_table_cmds = deepcopy(global_table_cmds)
                fq_cmd = blavg_table_cmds["AIPS FQ"]
                fq_cmd["keywords"] = blavg_fq_keywords
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

            # Now do some sanity checks of the merge UV file's metadata
            # against that of the blavg UV file. Mostly we check that the
            # frequency information is the same.
            merge_desc = merge_uvf.Desc.Dict
            merge_fq_keywords = dict(merge_uvf.tables["AIPS FQ"].keywords)
            merge_fq_rows = merge_uvf.tables["AIPS FQ"].rows

            # Compare
            # (1) UV FITS shape parameters
            # (2) FQ table keywords
            # (3) FQ table rows

            pprint({k: (merge_desc[k], blavg_desc[k]) for k in (
                        'inaxes', 'cdelt', 'crval', 'naxis', 'crota', 'crpix',
                         'ilocu', 'ilocv', 'ilocw', 'ilocb', 'iloct', 'ilocsu',
                        'jlocc', 'jlocs', 'jlocf', 'jlocif', 'jlocr', 'jlocd')
                        if not merge_desc[k] == blavg_desc[k]})

            assert all(merge_desc[k] == blavg_desc[k] for k in (
                        'inaxes', 'cdelt', 'crval', 'naxis', 'crota', 'crpix',
                         'ilocu', 'ilocv', 'ilocw', 'ilocb', 'iloct', 'ilocsu',
                        'jlocc', 'jlocs', 'jlocf', 'jlocif', 'jlocr', 'jlocd'))
            assert all(merge_fq_keywords[k] == blavg_fq_keywords[k] for k in blavg_fq_keywords.keys())
            assert len(merge_fq_rows) == len(blavg_fq_rows)
            assert all(all(mr[k] == br[k] for k in br.keys()) for mr, br in zip(merge_fq_rows, blavg_fq_rows))

            merge_uvf.append_history(blavg_history)

            # Record the starting visibility
            # for this scan in the merge file
            nx_row['START VIS'] = [merge_firstVis]

            log.info("Merging '%s' into '%s'", blavg_path, self.uv_merge_path)

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

                # Update starting positions
                blavg_firstVis += numVisBuff
                merge_firstVis += numVisBuff

            # Record the ending visibility
            # for this scan in the merge file
            nx_row['END VIS'] = [merge_firstVis-1]

            # Append row to index table
            merge_uvf.tables["AIPS NX"].rows.append(nx_row)

            # Remove scan and baseline averaged files once merged
            if 'scans' in self.clobber:
                log.info("Zapping '%s'", scan_uvf.aips_path)
                scan_uvf.Zap()

            if 'avgscans' in self.clobber:
                log.info("Zapping '%s'", blavg_uvf.aips_path)
                blavg_uvf.Zap()

        # Write the index table
        merge_uvf.tables["AIPS NX"].write()

        # Create an empty calibration table
        merge_uvf.attach_CL_from_NX_table(self.ka.max_antenna_number)

        # Close merge file
        merge_uvf.close()

