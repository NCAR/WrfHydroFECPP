from pathlib import Path

import ESMF
from mpi4py import MPI

from fecpp.app import CoastalPreprocessorApp
from fecpp.slp import SeaLevelPressure

comm = MPI.COMM_WORLD
assert comm.Get_size() == ESMF.pet_count()

if ESMF.local_pet() == 0:
    print("Creating grids", flush=True)
app = CoastalPreprocessorApp(Path('.'), Path('./geo_em.d01.conus_1km_NWMv2.1.nc'))

if ESMF.local_pet() == 0:
    print("Processing file(s)", flush=True)
app.regrid_to_lat_lon(input_file=Path('nwm.t00z.medium_range.forcing.f001.conus.nc'),
                      output_path_transformer=lambda x:x.with_suffix(".latlon.nc"),
                      var_filter=SeaLevelPressure)
