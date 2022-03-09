from pathlib import Path
import os

import ESMF
from mpi4py import MPI

from fecpp.app import CoastalPreprocessorApp
from fecpp.slp import SeaLevelPressure


def main():
    comm = MPI.COMM_WORLD
    comm.Set_errhandler(MPI.ERRORS_ARE_FATAL)
    assert comm.Get_size() == ESMF.pet_count()

    ESMF.Manager(debug=True)

    # find input forcings
    if int(os.environ.get("LENGTH_HRS", 0)) < 0:                 # AnA configuration
        dir_date = os.environ["FORCING_END_DATE"]
    else:
        dir_date = os.environ["FORCING_BEGIN_DATE"]
    if len(dir_date) == 12:
        # remove minutes
        dir_date = dir_date[:-2]

    input_path = Path(os.environ["NWM_FORCING_OUTPUT_DIR"]) / dir_date
    if ESMF.local_pet() == 0:
        print(f"Starting FECPP in {input_path}")

    output_path = Path(os.environ["COASTAL_FORCING_OUTPUT_DIR"])

    schism_mesh = Path(os.environ["SCHISM_ESMFMESH"])

    app = CoastalPreprocessorApp(input_path, output_path, Path(os.environ["GEOGRID_FILE"]), schism_mesh=schism_mesh)
    app.regrid_all_files(output_path_transformer=lambda x: output_path / (x.stem + ".latlon.nc"),
                         var_filter=SeaLevelPressure,
                         file_filter="**/*LDASIN_DOMAIN*")


if __name__ == '__main__':
    main()

