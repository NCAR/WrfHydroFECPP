import os
from pathlib import Path
import math
from threading import Thread
from queue import Queue

import ESMF
import numpy as np
from netCDF4 import Dataset, default_fillvals

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm.Set_errhandler(MPI.ERRORS_ARE_FATAL)


class CoastalPreprocessorApp(object):
    def __init__(self, input_dir: Path, output_dir: Path, geo_em_path: Path, schism_mesh: Path):
        self.geo = geo_em_path
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.job_idx = os.environ.get("FECPP_JOB_INDEX")
        self.job_count = os.environ.get("FECPP_JOB_COUNT")

        self.root = ESMF.local_pet() == 0
        self.schism_mesh = ESMF.Mesh(filename=str(schism_mesh), filetype=ESMF.FileFormat.ESMFMESH)
        self.schism_first_timestep = None
        self.schism_prev_time = -math.inf
        with Dataset(schism_mesh, 'r') as smesh:
            self.total_elements = smesh.dimensions['elementCount'].size

        node_coords = self.schism_mesh.coords[0]
        self.out_lat_range, self.out_lon_range = self._find_lat_lon_ranges_(node_coords)

        ds = Dataset(self.geo, 'r')
        self.src_height = ds.variables['HGT_M'][:]
        self._setup_hydro_grid_(ds)
        self._setup_latlon_grid_(ds)
        ds.close()

        self._regridder = None          # placeholders
        self._schism_regridder = None
        self.times = []

    def regrid_all_files(self, output_path_transformer, var_filter=None, file_filter=None):
        input_files = sorted(self.input_dir.glob(file_filter))
        file_zero = input_files[0]
        if self.job_idx is not None:
            idx = int(self.job_idx)
            jobs = int(self.job_count)
            count = math.ceil(len(input_files) / jobs)
            sub_input_files = input_files[idx*count: idx*count+count]
        else:
            idx = 0
            sub_input_files = input_files

        if self.root:
            input_ds = Dataset(file_zero)

            # TODO: we really should read actual time unit from metadata using udunits, etc.            
            if 'time' in input_ds.variables:
                start_time = input_ds['time'][0] * 60        # in minutes
            if 'valid_time' in input_ds.variables:
                start_time = input_ds['valid_time'][0]       # in seconds

            self.schism_first_timestep = start_time
            input_ds.close()

            # open netCDF vsource file and create header (if we're idx=0)
            if idx == 0:
                self.schism_vsource = Dataset(Path(self.output_dir / "precip_source.nc"), 'w', format="NETCDF4")
                self._build_source_nc(self.schism_vsource, ntimes=len(input_files), elems=np.arange(1, self.total_elements+1))

        for file in input_files:
            if file in sub_input_files:             # only process our task's subset
                if self.root:
                    print(f"Post-processing file: {file}", flush=True)
                self.regrid_to_lat_lon(file, output_path_transformer=output_path_transformer, var_filter=var_filter)
            if idx == 0:                            # only do precip regridding on a single job array task
                self.regrid_to_schism(file, output_path_transformer=output_path_transformer, var_filter=var_filter)

        if idx == 0 and self.root:
            self.schism_vsource.sync()
            self.schism_vsource.close()

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
        in_field.data[...] = nc_var[0, :].T[x_lbounds:x_ubounds, y_lbounds:y_ubounds]

        out_field = ESMF.Field(grid=self.schism_mesh, meshloc=ESMF.MeshLoc.ELEMENT, name=f"rainrate-out")
        out_field.data[...] = -1

        if self.root:
            print(f"Regridding output field `{nc_var.name}`"
                  f"{' (generating initial spatial weights)' if not self._schism_regridder else ''}", flush=True)

        if self._schism_regridder is None:
            self._schism_regridder = ESMF.Regrid(srcfield=in_field, dstfield=out_field,
                                                 regrid_method=ESMF.RegridMethod.BILINEAR,
                                                 unmapped_action=ESMF.UnmappedAction.IGNORE,
                                                 extrap_method=ESMF.ExtrapMethod.NONE)

        out_field = self._schism_regridder(srcfield=in_field, dstfield=out_field)  # , zero_region=ESMF.constants.Region.SELECT)
        out_field.data[...] = np.where(out_field.data[...] > 0, out_field.data[...], 0)

        # convert to volumetric flux (m^3/s)
        R0_SCHISM = 6378206.4           # earth radius in meters used by SCHISM
        DENSITY_FACTOR = 1000

        unit_areas = ESMF.Field(self.schism_mesh, meshloc=ESMF.MeshLoc.ELEMENT, name='areafield')
        unit_areas.get_area()
        areas_m2 = unit_areas.data[...] * (R0_SCHISM * R0_SCHISM)
        out_field.data[...] *= (areas_m2 / DENSITY_FACTOR)

        #   get element counts
        local_element_count = np.asarray([self.schism_mesh.size[1]], dtype='i')
        global_element_counts = np.empty((ESMF.pet_count(),), dtype='i')
        comm.Gather(local_element_count, global_element_counts)

        #   collect data from each rank
        if self.root:
            self.total_elements = global_element_counts.sum()
            all_elements = np.zeros((self.total_elements,))
        else:
            all_elements = None

        comm.Gatherv(sendbuf=out_field.data, recvbuf=(all_elements, global_element_counts))

        #   write field data to `vsource.th.2`
        if self.root:
            # TODO: we really should read actual time unit from metadata using udunits, etc.       
            if 'time' in input_ds.variables:
                step_time = input_ds['time'][0] * 60        # in minutes
            if 'valid_time' in input_ds.variables:
                step_time = input_ds['valid_time'][0]       # in seconds

            print(f"Writing vsource for step_time={step_time}")

            # # assert step_time > self.schism_prev_time, "ERROR: forcings time not increasing monotonically"
            # # print(f"self.schism_prev_time = {self.schism_prev_time},  step_time = {step_time}", flush=True)
            # self.schism_prev_time = step_time

            output_ts = int(step_time - self.schism_first_timestep)
            output_idx = (output_ts / 3600) - 1
            self.schism_vsource['time_vsource'][output_idx] = output_ts
            self.schism_vsource['vsource'][output_idx, :] = all_elements
            self.schism_vsource.sync()

        # print(f"FECPP wrote {(self.times[-1] - self.times[0]) / 3600.0 } hours of data to vsource.th", flush=True)

        # comm.Barrier()
        in_field.destroy()
        out_field.destroy()
        input_ds.close()

    def regrid_to_lat_lon(self, input_file: Path, output_path_transformer, var_filter=None):
        ll_vars = ['U2D', 'V2D', 'LWDOWN', 'T2D', 'Q2D', 'PSFC', 'SWDOWN', 'LQFRAC']

        input_ds = Dataset(input_file)

        nlons, nlats = self.out_grid.max_index

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

            in_field.data[...] = nc_var[0, :].T[x_lbounds:x_ubounds, y_lbounds:y_ubounds]

            out_field = ESMF.Field(grid=self.out_grid, name=f"{variable}-out")
            out_field.data[...] = 0.0  # default_fillvals['f4']

            if self.root:
                print(f"Regridding output field `{nc_var.name}`"
                      f"{' (generating initial spatial weights)' if not self._regridder else ''}", flush=True)

            if self._regridder is None:
                self._regridder = ESMF.Regrid(srcfield=in_field, dstfield=out_field,
                                              regrid_method=ESMF.RegridMethod.BILINEAR,
                                              unmapped_action=ESMF.UnmappedAction.IGNORE)
                                              # extrap_method=ESMF.ExtrapMethod.NEAREST_IDAVG)
            else:
                self._regridder(srcfield=in_field, dstfield=out_field, zero_region=ESMF.constants.Region.SELECT)

            # assemble output field
            global_output = np.zeros((nlons, nlats))
            y_lbounds, y_ubounds, x_lbounds, x_ubounds = self.out_bounds
            global_output[x_lbounds:x_ubounds, y_lbounds:y_ubounds] = out_field.data[...]

            final_output = np.empty((nlons, nlats))
            comm.Reduce(global_output.data, final_output.data)
            # comm.Barrier()

            if self.root:
                output_ds.variables[nc_var.name][0, :] = final_output.T

            in_field.destroy()
            out_field.destroy()

        if self.root:
            lat_coord[:] = self.lats
            lon_coord[:] = self.lons
            time_coord[:] = in_time[:]
            output_ds.close()
        input_ds.close()
        
    def make_schism_aux_inputs(self):
        if self.job_idx is None or int(self.job_idx) == 0:
            with open(self.output_dir / 'source_sink.in.2', 'w') as o:
                o.write(f"{self.total_elements}\n")
                o.write('\n'.join(map(str, range(1, self.total_elements+1))))
                o.write('\n'+'0')
                o.flush()

        suffix = "" if self.job_idx is None else f".part{int(self.job_idx):04}"
        line_one = '\t'.join(["-9999"] * self.total_elements)
        line_two = '\t'.join(["0"] * self.total_elements)
        with open(self.output_dir / ('msource.th.2' + suffix), 'w') as o:
            for i in range(len(self.times)):
                o.write(f"{self.times[i]}\t")
                o.write(line_one)
                o.write('\n')
                o.write(line_two)
                o.write('\n')
            o.flush()

    # INTERNAL METHODS

    def _setup_hydro_grid_(self, ds: Dataset):
        xlat = ds.variables['XLAT_M'][0, :].T
        xlon = ds.variables['XLONG_M'][0, :].T
        clat = ds.variables['XLAT_C'][0, :].T
        clon = ds.variables['XLONG_C'][0, :].T

        self.in_grid, self.in_bounds = self._build_grid_(xlat, xlon, clat, clon)

    def _setup_latlon_grid_(self, ds: Dataset):
        xlat = ds.variables['XLAT_V']
        xlon = ds.variables['XLONG_U']
        xlat = ds.variables['XLAT_M']
        xlon = ds.variables['XLONG_M']

        # grid_factor = 1
        # nlat, nlon = map(lambda x: x*grid_factor, xlat.shape[1:])

        # determine a naive lat/lon overlay
        # lat_min, lat_max = xlat[:].min(), xlat[:].max()
        # lon_min, lon_max = xlon[:].min(), xlon[:].max()

        lat_min, lat_max = self.out_lat_range
        lon_min, lon_max = self.out_lon_range

        # self.lats, d_lat = np.linspace(lat_min, lat_max, nlat, retstep=True)
        # self.lons, d_lon = np.linspace(lon_min, lon_max, nlon, retstep=True)
        dlat = dlon = 0.01
        self.lats = np.arange(math.floor(lat_min/dlat)*dlat, (math.ceil(lat_max/dlat)*dlat)+dlat, dlat)
        self.lons = np.arange(math.floor(lon_min/dlon)*dlon, (math.ceil(lon_max/dlon)*dlon)+dlon, dlon)

        longitudes, latitudes = np.meshgrid(self.lons, self.lats, indexing="ij")

        self.out_grid, self.out_bounds = self._build_grid_(latitudes, longitudes)

    @staticmethod
    def _find_lat_lon_ranges_(node_coords):
        g_lon_min = np.empty([1], dtype=np.float32)
        g_lon_max = np.empty([1], dtype=np.float32)
        g_lat_min = np.empty([1], dtype=np.float32)
        g_lat_max = np.empty([1], dtype=np.float32)

        comm.Allreduce(np.float32(min(node_coords[0])), g_lon_min, op=MPI.MIN)
        comm.Allreduce(np.float32(max(node_coords[0])), g_lon_max, op=MPI.MAX)
        comm.Allreduce(np.float32(min(node_coords[1])), g_lat_min, op=MPI.MIN)
        comm.Allreduce(np.float32(max(node_coords[1])), g_lat_max, op=MPI.MAX)

        return (g_lat_min[0], g_lat_max[0]), (g_lon_min[0], g_lon_max[0])

    @staticmethod
    def _build_grid_(latitudes: np.ndarray, longitudes: np.ndarray, corner_lats: np.ndarray = None, corner_lons: np.ndarray = None):
        lon, lat = 0, 1  # Labels for coordinate axes
        nlon, nlat = latitudes.shape

        stagger = [ESMF.StaggerLoc.CENTER, ESMF.StaggerLoc.CORNER] if corner_lats is not None else [ESMF.StaggerLoc.CENTER]
        # noinspection PyTypeChecker
        grid = ESMF.Grid(np.array([nlon, nlat]),
                         staggerloc=stagger,
                         coord_sys=ESMF.CoordSys.SPH_DEG)

        y_lbounds = grid.lower_bounds[ESMF.StaggerLoc.CENTER][lat]
        y_ubounds = grid.upper_bounds[ESMF.StaggerLoc.CENTER][lat]
        x_lbounds = grid.lower_bounds[ESMF.StaggerLoc.CENTER][lon]
        x_ubounds = grid.upper_bounds[ESMF.StaggerLoc.CENTER][lon]

        if corner_lons is not None and corner_lats is not None:
            yc_lbounds = grid.lower_bounds[ESMF.StaggerLoc.CORNER][lat]
            yc_ubounds = grid.upper_bounds[ESMF.StaggerLoc.CORNER][lat]
            xc_lbounds = grid.lower_bounds[ESMF.StaggerLoc.CORNER][lon]
            xc_ubounds = grid.upper_bounds[ESMF.StaggerLoc.CORNER][lon]

        # print(f"[{ESMF.local_pet()+1} of {ESMF.pet_count()}] =>", x_lbounds, x_ubounds, y_lbounds, y_ubounds)

        lon_coord = grid.get_coords(lon)
        lon_par = longitudes[x_lbounds:x_ubounds, y_lbounds:y_ubounds]
        lon_coord[...] = lon_par

        if corner_lons is not None:
            lon_corners = grid.get_coords(lon, staggerloc=ESMF.StaggerLoc.CORNER)
            lonc_par = corner_lons[xc_lbounds:xc_ubounds, yc_lbounds:yc_ubounds]
            lon_corners[...] = lonc_par

        lat_coord = grid.get_coords(lat)
        lat_par = latitudes[x_lbounds:x_ubounds, y_lbounds:y_ubounds]
        lat_coord[...] = lat_par

        if corner_lats is not None:
            lat_corners = grid.get_coords(lat, staggerloc=ESMF.StaggerLoc.CORNER)
            latc_par = corner_lats[xc_lbounds:xc_ubounds, yc_lbounds:yc_ubounds]
            lat_corners[...] = latc_par

        return grid, (y_lbounds, y_ubounds, x_lbounds, x_ubounds)

    @staticmethod
    def _build_source_nc(ds, ntimes, elems):
        nelems = len(elems)

        # dimensions
        ds.createDimension('time_vsource', ntimes)
        ds.createDimension('nsources', nelems)

        ds.createDimension('one', 1)

        # variables
        eso = ds.createVariable('source_elem', 'i4', ('nsources',))
        _ = ds.createVariable('vsource', 'f8', ('time_vsource', 'nsources',), zlib=True)
        _ = ds.createVariable('time_vsource', 'f8', ('time_vsource',))
        vts = ds.createVariable('time_step_vsource', 'f4', ('one',))

        eso[:] = elems
        vts[:] = 3600
