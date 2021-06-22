from pathlib import Path

import ESMF
import mpi4py
import numpy as np
from netCDF4 import Dataset, default_fillvals


comm = mpi4py.MPI.COMM_WORLD


class CoastalPreprocessorApp(object):
    def __init__(self, input_dir: Path, geo_em_path: Path):
        self.geo = geo_em_path
        self.input_dir = input_dir

        self.root = ESMF.local_pet() == 0

        ds = Dataset(self.geo, 'r')
        self.src_height = ds.variables['HGT_M'][:]
        self._setup_hydro_grid_(ds)
        self._setup_latlon_grid_(ds)
        ds.close()

        self.regridder = None       # placeholder

    def _setup_hydro_grid_(self, ds: Dataset):
        xlat = ds.variables['XLAT_M'][0, :]
        xlon = ds.variables['XLONG_M'][0, :]

        self.in_grid, self.in_bounds = self.__build_grid__(xlat, xlon)

    def _setup_latlon_grid_(self, ds: Dataset):
        # xlat = ds.variables['XLAT_V']
        # xlon = ds.variables['XLONG_U']
        xlat = ds.variables['XLAT_M']
        xlon = ds.variables['XLONG_M']

        grid_factor = 1
        nlat, nlon = map(lambda x: x*grid_factor, xlat.shape[1:])

        # determine a naive lat/lon overlay
        lat_min, lat_max = xlat[:].min(), xlat[:].max()
        lon_min, lon_max = xlon[:].min(), xlon[:].max()

        self.lats, d_lat = np.linspace(lat_min, lat_max, nlat, retstep=True)
        self.lons, d_lon = np.linspace(lon_min, lon_max, nlon, retstep=True)

        latitudes, longitudes = np.meshgrid(self.lats, self.lons, indexing="ij")

        self.out_grid, self.out_bounds = self.__build_grid__(latitudes, longitudes)

    def regrid_all_files(self, output_path_transformer, var_filter=None, file_filter=None):
        input_files = self.input_dir.glob(file_filter)
        for file in input_files:
            if self.root:
                print(f"Post-processing file: {file}", flush=True)
            self.regrid_to_lat_lon(file, output_path_transformer=output_path_transformer, var_filter=var_filter)

    def regrid_to_lat_lon(self, input_file: Path, output_path_transformer, var_filter=None):
        hydro_vars = ['U2D', 'V2D', 'LWDOWN', 'RAINRATE', 'T2D', 'Q2D', 'PSFC', 'SWDOWN', 'LQFRAC']

        input_ds = Dataset(input_file)

        nlats, nlons = self.out_grid.max_index

        if self.root:
            output_ds = Dataset(output_path_transformer(input_file), 'w')

            output_ds.createDimension(dimname="lat", size=nlats)
            output_ds.createDimension(dimname="lon", size=nlons)
            output_ds.createDimension(dimname="time", size=0)

            output_ds.createVariable(varname="lat", dimensions=("lat",), datatype=self.lats.dtype)
            lat_coord = output_ds.variables['lat']
            lat_coord.long_name = 'latitude'
            lat_coord.units = 'degrees_north'
            lat_coord.standard_name = 'latitude'
            lat_coord.axis = 'Y'

            output_ds.createVariable(varname="lon", dimensions=("lon",), datatype=self.lons.dtype)
            lon_coord = output_ds.variables['lon']
            lon_coord.long_name = 'longitude'
            lon_coord.units = 'degrees_east'
            lon_coord.standard_name = 'longitude'
            lon_coord.axis = 'X'

            in_time_name = {'time', 'valid_time'}.intersection({v for v in input_ds.variables}).pop()
            in_time = input_ds.variables[in_time_name]
            output_ds.createVariable(varname="time", dimensions=("time",), datatype=in_time.datatype)
            time_coord = output_ds.variables['time']
            time_coord.long_name = 'valid output time'
            time_coord.units = in_time.units
            time_coord.calendar = 'standard'
            time_coord.standard_name = 'time'

        else:
            output_ds = lat_coord = lon_coord = in_time = time_coord = None

        for variable in hydro_vars:
            if variable not in input_ds.variables:
                continue

            nc_var = input_ds.variables[variable]
            if var_filter is not None:
                original_name = nc_var.name
                nc_var = var_filter(nc_var, height=self.src_height)
                if original_name != nc_var.name:
                    if self.root:
                        print(f"Transforming variable `{original_name}` to `{nc_var.name}`")

            if self.root:
                new_var = output_ds.createVariable(varname=nc_var.name,
                                                   datatype='f4',
                                                   dimensions=("time", "lat", "lon"))
                if 'standard_name' in nc_var.ncattrs():
                    new_var.standard_name = nc_var.standard_name
                if 'long_name' in nc_var.ncattrs():
                    new_var.long_name = nc_var.long_name
                if 'units' in nc_var.ncattrs():
                    new_var.units = nc_var.units

            in_field = ESMF.Field(grid=self.in_grid, name=f"{variable}-in")
            y_lbounds, y_ubounds, x_lbounds, x_ubounds = self.in_bounds
            in_field.data[...] = nc_var[0, y_lbounds:y_ubounds, x_lbounds:x_ubounds]

            out_field = ESMF.Field(grid=self.out_grid, name=f"{variable}-out")
            out_field.data[...] = default_fillvals['f4']

            if self.root:
                print(f"Regridding output field `{nc_var.name}`"
                      f"{' (generating initial spatial weights)' if not self.regridder else ''}", flush=True)

            if self.regridder is None:
                self.regridder = ESMF.Regrid(srcfield=in_field, dstfield=out_field,
                                             regrid_method=ESMF.RegridMethod.PATCH,
                                             unmapped_action=ESMF.UnmappedAction.IGNORE,
                                             line_type=ESMF.LineType.GREAT_CIRCLE,
                                             extrap_method=ESMF.api.constants.ExtrapMethod.NEAREST_STOD)
            else:
                self.regridder(srcfield=in_field, dstfield=out_field, zero_region=ESMF.constants.Region.SELECT)

            # assemble output field
            global_output = np.zeros((nlats, nlons))
            y_lbounds, y_ubounds, x_lbounds, x_ubounds = self.out_bounds
            global_output[y_lbounds:y_ubounds, x_lbounds:x_ubounds] = out_field.data[...]

            final_output = np.empty((nlats, nlons))
            comm.Reduce(global_output.data, final_output.data)
            # comm.Barrier()

            if self.root:
                output_ds.variables[nc_var.name][0, :] = final_output

            in_field.destroy()
            out_field.destroy()

        if self.root:
            lat_coord[:] = self.lats
            lon_coord[:] = self.lons
            time_coord[:] = in_time[:]
            output_ds.close()

    # INTERNAL METHODS

    @staticmethod
    def __build_grid__(latitudes: np.ndarray, longitudes: np.ndarray):
        lat, lon = 0, 1  # Labels for coordinate axes

        nlat, nlon = latitudes.shape

        # noinspection PyTypeChecker
        grid = ESMF.Grid(np.array([nlat, nlon]),
                         staggerloc=[ESMF.StaggerLoc.CENTER],
                         coord_sys=ESMF.CoordSys.SPH_DEG)

        y_lbounds = grid.lower_bounds[ESMF.StaggerLoc.CENTER][lat]
        y_ubounds = grid.upper_bounds[ESMF.StaggerLoc.CENTER][lat]
        x_lbounds = grid.lower_bounds[ESMF.StaggerLoc.CENTER][lon]
        x_ubounds = grid.upper_bounds[ESMF.StaggerLoc.CENTER][lon]

        # print(f"[{ESMF.local_pet()+1} of {ESMF.pet_count()}] =>", x_lbounds, x_ubounds, y_lbounds, y_ubounds)

        lon_coord = grid.get_coords(lon)
        lon_par = longitudes[y_lbounds:y_ubounds, x_lbounds:x_ubounds]
        lon_coord[...] = lon_par

        lat_coord = grid.get_coords(lat)
        lat_par = latitudes[y_lbounds:y_ubounds, x_lbounds:x_ubounds]
        lat_coord[...] = lat_par

        return grid, (y_lbounds, y_ubounds, x_lbounds, x_ubounds)
