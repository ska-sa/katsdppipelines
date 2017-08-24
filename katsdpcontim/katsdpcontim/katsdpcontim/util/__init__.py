import ast
import logging

log = logging.getLogger('katsdpcontim')

def parse_katdal_select(select_str):
    """
    Parses a string, containing assign statements
    that will turned be into kwargs suitable for
    passing to :meth:`katdal.DataSet.select`.

    .. code-block:: python

        h5 = katdal.open('123456789.h5')
        kwargs = parse_katdal_select("spw=3; scans=[1,2];
                                      targets='bpcal,radec'")
        h5.select(**kwargs)

    Parameters
    ----------
    select_str: str
        Selection string. Should only contain
        assignment statements assigning
        python literal values to names,
        separated by semi-colons.

    Returns
    -------
    dict
        Dictionary { name: value } containing
        assignment results.
    """

    if not select_str:
        return {}

    try:
        return { target.id: ast.literal_eval(stmt.value)
                for stmt in ast.parse(select_str, mode='single').body
                for target in stmt.targets}
    except SyntaxError as e:
        log.exception("Exception parsing katdal selection string "
                    "'{}'".format(select_str))
        raise e
