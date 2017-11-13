import argparse
from pretty import pprint

from katsdpcontim import obit_context
import katsdpcontim.uv_facade as uvf

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="AIPS name")
    parser.add_argument("-c", "--class", default="raw", dest="aclass",
                                        help="AIPS class")
    parser.add_argument("-d", "--disk", default=1,
                                        type=int,
                                        help="AIPS disk")
    parser.add_argument("-s", "--seq", default=1,
                                        type=int,
                                        help="AIPS sequence")

    return parser

args = create_parser().parse_args()

with obit_context():
    uv = uvf.open_uv(args.name, args.disk,
                    args.aclass, args.seq, dtype="AIPS", mode="r")
#     uv.attach_table("AIPS FG", 1)

#     pprint(uv.tables["AIPS AN"]._table.IODesc.List.Dict)
#     pprint(uv.tables["AIPS AN"]._table.Desc.Dict)
#     pprint(uv.tables["AIPS AN"].keywords['ARRAYX'])
#     uv.tables["AIPS AN"].keywords['ARRAYX'] = 1.0
#     pprint(uv.tables["AIPS AN"].keywords['ARRAYX'])
#     uv.tables["AIPS AN"].keywords['ARRNAM'] = "ADSLGKADSGLKAGASDGLK"
#     pprint(uv.tables["AIPS AN"].keywords.update({"ARRAYX": 1.0, "ARRAYY": 1.0}))
#     pprint(uv.tables["AIPS AN"].keywords.asdict)
#     pprint(uv.tables["AIPS AN"].fields['ANNAME'])
#     pprint(uv.tables["AIPS AN"].name)
#     pprint(uv.tables["AIPS AN"].default_row)
#     #pprint(uv.tables["AIPS AN"].rows)
#     pprint(uv.tables["AIPS FQ"]._table.Desc.Dict)

#     pprint(uv.tables["AIPS AN"].rows[0])
#     uv.tables["AIPS AN"].rows[0]['ANNAME'] = "BOB"
#     uv.tables["AIPS AN"].rows.write()
# #    uv.tables["AIPS AN"].rows.read()
#     pprint(uv.tables["AIPS CL"].rows)
#     pprint(uv.tables["AIPS CL"].nrow)
#     uv.tables["AIPS CL"].rows.write()
#     uv.tables["AIPS CL"].rows.read()

    tables = [(n, t) for n, t in uv.tables.items()
                if n in ["AIPS CL"]]


    for name, table in tables:
        pprint(["%s rows" % name, table.rows])
        pprint(["%s keywords" % name, table.keywords])
        pprint(["%s defaults" % name, table.default_row])

    pprint(uv._uv.Desc.Dict['nvis'])

    uv.close()


