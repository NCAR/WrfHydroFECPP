from pathlib import Path
import math

import ESMF
import numpy as np
from netCDF4 import Dataset, default_fillvals
from os import path
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm.Set_errhandler(MPI.ERRORS_ARE_FATAL)

class CoastalPreprocessorApp(object):
    def __init__(self, input_dir: Path, geo_em_path: Path, schism_mesh: Path):
        if (ESMF.version_compare(ESMF.__version__,'8.1.0') < 0):
            raise('ESMF version needs to be 8.1.0 or greater. Current version is ',
                  ESMF.__version__)
        self.geo = geo_em_path
        self.input_dir = input_dir

        self.root = ESMF.local_pet() == 0

        ds = Dataset(self.geo, 'r')
        self.src_height = ds.variables['HGT_M'][:]
        self._setup_hydro_grid_(ds)
        self._setup_latlon_grid_(ds)
        ds.close()

        self.schism_mesh = ESMF.Mesh(filename=str(schism_mesh), filetype=ESMF.FileFormat.ESMFMESH)
        self.schism_vsource = open(Path(input_dir / 'vsource.th.2'), 'w')
        self.schism_first_timestep = None
        self.schism_prev_time = -math.inf

        self._regridder = None          # placeholders
        self._schism_regridder = None
        self.total_elements = 0
        self.times = []

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

        # self.lats, d_lat = np.linspace(lat_min, lat_max, nlat, retstep=True)
        # self.lons, d_lon = np.linspace(lon_min, lon_max, nlon, retstep=True)
        dlat = dlon = 0.01
        self.lats = np.arange(math.floor(lat_min), math.ceil(lat_max)+dlat, dlat)
        self.lons = np.arange(math.floor(lon_min), math.ceil(lon_max)+dlon, dlon)

        latitudes, longitudes = np.meshgrid(self.lats, self.lons, indexing="ij")

        self.out_grid, self.out_bounds = self.__build_grid__(latitudes, longitudes)

    def regrid_all_files(self, output_path_transformer, var_filter=None, file_filter=None):
        sorted_input_files = sorted(self.input_dir.glob(file_filter))
        num_input_files = len(sorted_input_files)
        for i, file in enumerate(sorted_input_files, start=1):
            if self.root:
                print(f"Post-processing file: {file}", flush=True)
            self.regrid_to_lat_lon(file, output_path_transformer=output_path_transformer, var_filter=var_filter)
            self.regrid_to_schism(file, output_path_transformer=output_path_transformer, var_filter=var_filter)

            # moved comm.Barrier() from self.regrid_to_schism() so make_schism_aux_inputs() can run
            #   while last input is being writter
            if (i == num_input_files):
                if (ESMF.pet_count() > 1) and (ESMF.local_pet() == 1):
                    self.make_schism_aux_inputs()
                elif (ESMF.pet_count() == 1) and (self.root):
                    self.make_schism_aux_inputs()
            comm.Barrier()

    def regrid_to_schism(self, input_file: Path, output_path_transformer, var_filter=None):
        input_ds = Dataset(input_file)

        nc_var = input_ds.variables['RAINRATE']
        if var_filter is not None:
            original_name = nc_var.name
            nc_var = var_filter(nc_var, height=self.src_height)
            if original_name != nc_var.name:
                if self.root:
                    print(f"Transforming variable `{original_name}` to `{nc_var.name}`")

        in_field = ESMF.Field(grid=self.in_grid, name=f"rainrate-in")
        y_lbounds, y_ubounds, x_lbounds, x_ubounds = self.in_bounds
        in_field.data[...] = nc_var[0, y_lbounds:y_ubounds, x_lbounds:x_ubounds]

        out_field = ESMF.Field(grid=self.schism_mesh, meshloc=ESMF.MeshLoc.ELEMENT, name=f"rainrate-out")
        out_field.data[...] = -1
        out_field.get_area()

        if self.root:
            print(f"Regridding output field `{nc_var.name}`"
                    f"{' (generating initial spatial weights)' if not self._schism_regridder else ''}", flush=True)

        if self._schism_regridder is None:
            self._schism_regridder = ESMF.Regrid(srcfield=in_field, dstfield=out_field,
                                                regrid_method=ESMF.RegridMethod.BILINEAR,
                                                unmapped_action=ESMF.UnmappedAction.IGNORE,
                                                extrap_method=ESMF.ExtrapMethod.CREEP_FILL)

        out_field = self._schism_regridder(srcfield=in_field, dstfield=out_field) #, zero_region=ESMF.constants.Region.SELECT)

        # TEXT OUTPUT STAGE

        #   get element counts
        local_element_count = np.asarray([self.schism_mesh.size[1]], dtype='i')
        global_element_counts = np.empty((ESMF.pet_count(),), dtype='i')
        comm.Gather(local_element_count, global_element_counts)

        #   collect data from each rank
        if self.root:
            self.total_elements = global_element_counts.sum()
            all_elements = np.empty((self.total_elements,))
        else:
            all_elements = None
        comm.Gatherv(sendbuf=out_field.data, recvbuf=(all_elements, global_element_counts))

        #   write field data to `vsource.th.2`
        if self.root:
            # TODO: currently these are in the input field in minutes, so convert to seconds
            # TODO: we really should read actual time unit from metadata using udunits, etc.
            step_time = input_ds['time'][0] * 60

            # assert step_time > self.schism_prev_time, "ERROR: forcings time not increasing monotonically"
            self.schism_prev_time = step_time
            if self.schism_first_timestep is None:
                self.schism_first_timestep = step_time

            output_ts = int(step_time - self.schism_first_timestep)
            self.times.append(output_ts)
            self.schism_vsource.write(f"{output_ts}")

            for i in range(all_elements.size):
                 self.schism_vsource.write(f" {all_elements[i]}")

            self.schism_vsource.write("\n")
            self.schism_vsource.flush()         # TODO: add a proper close() call after last timestep!

        # moved barrier outside of regrid_to_schism so make_schism_aux_inputs can be run in parallel
        # comm.Barrier()


    def regrid_to_lat_lon(self, input_file: Path, output_path_transformer, var_filter=None):
        ll_vars = ['U2D', 'V2D', 'LWDOWN', 'T2D', 'Q2D', 'PSFC', 'SWDOWN', 'LQFRAC']

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

        for variable in ll_vars:
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
                      f"{' (generating initial spatial weights)' if not self._regridder else ''}", flush=True)

            if self._regridder is None:
                self._regridder = ESMF.Regrid(srcfield=in_field, dstfield=out_field,
                                              regrid_method=ESMF.RegridMethod.BILINEAR,
                                              unmapped_action=ESMF.UnmappedAction.IGNORE,
                                              line_type=ESMF.LineType.GREAT_CIRCLE,
                                              extrap_method=ESMF.api.constants.ExtrapMethod.NEAREST_STOD)
            else:
                self._regridder(srcfield=in_field, dstfield=out_field, zero_region=ESMF.constants.Region.SELECT)

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

        input_ds.close()

    def make_schism_aux_inputs(self):
        sink_file = self.input_dir / 'source_sink.in.2'
        file_elements = open(sink_file).readline() if path.exists(sink_file) else -1
        if (int(file_elements) != self.total_elements):
            with open(sink_file, 'w') as o:
                o.write(f"{self.total_elements}\n")
                o.write('\n'.join(map(str, range(1, self.total_elements+1))))
                o.write('\n'+'0')

        with open(self.input_dir / 'msource.th.2', 'w') as o:
            for i in range(len(self.times)):
                o.write(f"{self.times[i]}\t")
                o.write('\t'.join(["-9999"] * self.total_elements))
                o.write('\n')
                o.write('\t'.join(["0"] * self.total_elements))
                o.write('\n')

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
