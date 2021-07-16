from pathlib import Path
import os

import ESMF
from mpi4py import MPI

from fecpp.app import CoastalPreprocessorApp
from fecpp.slp import SeaLevelPressure

def main():
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == ESMF.pet_count()

    ESMF.Manager(debug=True)

    # find input forcings
    dir_date = os.environ["FORCING_BEGIN_DATE"]
    if len(dir_date) == 12:
        # remove minutes
        dir_date = dir_date[:-2]
        
    input_path = Path(os.environ["FORCING_OUTPUT_DIR"]) / dir_date
    if ESMF.local_pet() == 0:
        print(f"Starting FECPP in {input_path}")
    
    schism_mesh = Path(os.environ["SCHISM_ESMFMESH"])

    app = CoastalPreprocessorApp(input_path, Path(os.environ["GEOGRID_FILE"]), schism_mesh=schism_mesh)
    app.regrid_all_files(output_path_transformer=lambda x:x.with_suffix(".latlon.nc"),
                         var_filter=SeaLevelPressure,
                         file_filter="**/*LDASIN_DOMAIN*")


if __name__ == '__main__':
    main()