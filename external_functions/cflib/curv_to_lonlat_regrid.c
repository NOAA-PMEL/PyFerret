/**
 * curv_to_lonlat_regrid
 * from tst_tripolar_to_lonlat_regrid from the cflib distribution
 */

#include <Python.h> /* make sure Python.h is first */
#include "nccf_regrid.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <netcdf.h>
#include <libcf_src.h>

#include <nccf_coord.h>
#include <nccf_grid.h>
#include <nccf_data.h>
#include <nccf_utility_functions.h>
#include <nccf_handle_error.h>

/*#include <config.h> */

void
setDataToFill(int nvertex, double FillValue, double data[]) {
  int k;
  for (k = 0; k < nvertex; ++k) {
    data[k] = FillValue;
  }
}

//////////////////////////////////////////////////////////////////////

void 
readCurvi(const int dims[],  const char *filename, const char *datavar,
		     int coordIds[], int *gridId, int *dataId){
  const int ndims = 2;
  int nvertex = dims[0]*dims[1];
  const int save = 1;
  int status;
  double *clon, *clat;
  double *data;
  int *imask;

#define RANK 3
#define D2D 2

/* Define whole variable start/count should dynamically allocate these; dims[0], dims[1]*/
static size_t start0[RANK] = {12,0,0};
static size_t count0[RANK] = {1,336,896};

/* Define 2D variable start/count */
static size_t start1[D2D] = {0,0};
static size_t count1[D2D] = {336,896};

  int ncid, varid;
  int i;
  size_t start[RANK];
  size_t count[RANK];

  const char *dimnames[] = {"nj", "ni"};

  clat = ( double* )FerMem_Malloc(nvertex * sizeof(double), __FILE__, __LINE__);
  clon = ( double* )FerMem_Malloc(nvertex * sizeof(double), __FILE__, __LINE__);
  data = ( double* )FerMem_Malloc(nvertex * sizeof(double), __FILE__, __LINE__);
  imask = ( int* )FerMem_Malloc(nvertex * sizeof(int), __FILE__, __LINE__);

    if((status = nc_open( filename, NC_NOWRITE, &ncid ))) ERR;

    if((status = nc_inq_varid(ncid, datavar, &varid))) ERR;

    /* Read one timestep */
    memcpy(start,start0,sizeof(start0));
    memcpy(count,count0,sizeof(count0));

    printf("*** reading 1 timestep\n");
    if((status = nc_get_vara_double(ncid,varid,start,count,(double*)data))) ERR;

    /* Read the mask and coordinate variables */
    memcpy(start,start1,sizeof(start1));
    memcpy(count,count1,sizeof(count1));

    printf("*** reading mask_rho\n");
    if((status = nc_inq_varid(ncid, "mask_rho", &varid))) ERR;
	
    if((status = nc_get_vara_int(ncid,varid,start,count,(int*)imask))) ERR;

    printf("*** reading lon coords\n");
    if((status = nc_inq_varid(ncid, "lon_rho", &varid))) ERR;
	
    if((status = nc_get_vara_double(ncid,varid,start,count,(double*)clon))) ERR;

    printf("*** reading lat coords\n");
    if((status = nc_inq_varid(ncid, "lat_rho", &varid))) ERR;

    if((status = nc_get_vara_double(ncid,varid,start,count,(double*)clat))) ERR;

  if ((status = nccf_def_lat_coord(ndims, dims, dimnames, clat, save, &coordIds[0]))) ERR;
  if ((status = nccf_def_lon_coord(ndims, dims, dimnames, clon, save, &coordIds[1]))) ERR;
  if ((status = nccf_def_grid(coordIds, "curvi_grid", gridId))) ERR;
  if ((status = nccf_save_grid_scrip(*gridId, "curvi_grid_scrip.nc"))) ERR;
  /* args 3,4,5 are	standard_name,units,time_dimname,  */
  if ((status = nccf_def_data(*gridId, datavar, NULL, NULL, NULL, dataId))) ERR;
  if ((status = nccf_set_grid_validmask(*gridId, imask ))) ERR;


  /* Set the data */
  if ((status = nccf_set_data_double(*dataId, data, save, 
					    NC_FILL_DOUBLE))) ERR;

  FerMem_Free(clat, __FILE__, __LINE__);
  FerMem_Free(clon, __FILE__, __LINE__);
  FerMem_Free(data, __FILE__, __LINE__);
  FerMem_Free(imask, __FILE__, __LINE__);
}

//////////////////////////////////////////////////////////////////////

void 
createLonLat(const double xymin[], const double xymax[], 
	     const int dims[], 
	     void (*setDataFunct)(int nv, double FillValue, double d[]),
	     double FillValue, int coordIds[], int *gridId, int *dataId){

  const int ndims = 2;
  int nvertex = dims[0]*dims[1];
  const int save = 1;
  int status;
  int i, j, k;
  double *clon, *clat;
  double *data;
  double dxs[ndims];

  const char *dimnames[] = {"nj", "ni"};

  clat = ( double* )FerMem_Malloc(nvertex * sizeof(double), __FILE__, __LINE__);
  clon = ( double* )FerMem_Malloc(nvertex * sizeof(double), __FILE__, __LINE__);
  data = ( double* )FerMem_Malloc(nvertex * sizeof(double), __FILE__, __LINE__);

  for (i = 0; i < ndims; ++i) {
    dxs[i] = (xymax[i] - xymin[i]) / (dims[i] - 1);
  }

  /* Populate coordinates and create lon/lat coordinate objects */
  for (j = 0; j < dims[0]; ++j) {
    for (i = 0; i < dims[1]; ++i) {
      k = i + dims[1]*j;
      clat[k] = xymin[0] + j*dxs[0];
      clon[k] = xymin[1] + i*dxs[1];
    }
  }  

  if ((status = nccf_def_lat_coord(ndims, dims, dimnames, clat, save, &coordIds[0]))) ERR;
  if ((status = nccf_def_lon_coord(ndims, dims, dimnames, clon, save, &coordIds[1]))) ERR;
  if ((status = nccf_def_grid(coordIds, "lonlat_grid", gridId))) ERR;
  if ((status = nccf_save_grid_scrip(*gridId, "lonlat_grid_scrip.nc"))) ERR;
  if ((status = nccf_def_data(*gridId, "data", NULL, NULL, NULL, dataId))) ERR;

  /* Set the data. the VALUE of the data on this grid will be the fillvalue of
     the regridded data. The output name of the variable also comes from the call above
	 so could pass the original variable name to nccf_def_data.*/
  setDataFunct(nvertex, FillValue, data);
  if ((status = nccf_set_data_double(*dataId, data, save, 
					    NC_FILL_DOUBLE))) ERR;

  FerMem_Free(clat, __FILE__, __LINE__);
  FerMem_Free(clon, __FILE__, __LINE__);
  FerMem_Free(data, __FILE__, __LINE__);
}


//////////////////////////////////////////////////////////////////////

void 
writeData(int gridId, int dataId, const char *filename){

  /* write lonlat data and coordinates to file */
  int ncid, status;
  if ((status = nc_create(filename, NC_CLOBBER, &ncid))) ERR;
  if ((status = nccf_put_grid(gridId, ncid))) ERR;
  if ((status = nccf_put_data(dataId, ncid))) ERR;
  if ((status = nc_close(ncid))) ERR;
}

//////////////////////////////////////////////////////////////////////

int main(){

  int status;
  const int ndims = 2;
  int i;
  
  int oriDims[] = {336,896}; 
  int ori_coord_ids[ndims], ori_grid_id, oriDataId;

  int tgtDims[] = {152, 315}; 
  int tgt_coord_ids[ndims], tgt_grid_id, tgtDataId;
  double xymin[] = {11., -102.0};
  double xymax[] = {49., -52.0};
  double FillValue = 9.96921E+36;

  readCurvi(oriDims, "http://geoport.whoi.edu:8081/thredds/dodsC/coawst_2_2/fmrc/coawst_2_2_best.ncd", 
	  "Lwave", ori_coord_ids, &ori_grid_id, &oriDataId);
  createLonLat(xymin, xymax, tgtDims, setDataToFill, FillValue,
	       tgt_coord_ids, &tgt_grid_id, &tgtDataId);

  writeData(ori_grid_id, oriDataId, "curv_to_lonlat_regrid_ori.nc");

#ifdef HAVE_LAPACK_LIB
  /* Create regrid object */
  int regrid_id;
  /* the lon coordinates are not periodic in the sense that 360 is a different
     coordinate value from 0, so no periodicity here */
  const int is_periodic[] = {0, 0};
  const int nitermax = 20;
  const double tolpos = 1.e-2;
  if ((status = nccf_def_regrid(ori_grid_id, tgt_grid_id, &regrid_id))) ERR;

  /* Exclude cut in longitude from searchable domain */
  /*  const int lo[] = {0, oriDims[1]/2 - 1}; */
  /*  const int hi[] = {oriDims[0]/2, oriDims[1]/2}; */
  /*  if ((status = nccf_add_regrid_forbidden(regrid_id, lo, hi))) ERR; */


    printf("*** computing weights\n");
  if ((status = nccf_compute_regrid_weights(regrid_id,
					    nitermax,
					    tolpos,
					    is_periodic))) ERR;

  
  /* Write the weights, indices and inside_domain vars */
  int ncid;
  const char* put_filename = "curv_to_lonlat_regrid_weights12.nc";

  if (( status = nc_create( put_filename, NC_CLOBBER, &ncid ))) ERR;
  if (( status = nccf_put_regrid(regrid_id, ncid)));
  if (( status = nc_close( ncid ))) ERR; 

  /* Interpolate */
  if ((status = nccf_apply_regrid(regrid_id, oriDataId, tgtDataId))) ERR;


  /* Write regrided data*/
  writeData(tgt_grid_id, tgtDataId, "curv_to_lonlat_regrid_intrp.nc");

  /* Check */
  int nvalid, ntargets;
  if ((status = nccf_inq_regrid_ntargets(regrid_id, &ntargets))) ERR;
  if ((status = nccf_inq_regrid_nvalid(regrid_id, &nvalid))) ERR;
  double ratio = (double)(nvalid) / (double)(ntargets);
  printf("ratio of valid to num target points = %f\n", ratio);

  /* Clean up */
  if ((status = nccf_free_regrid(regrid_id))) ERR;
#endif


  if ((status = nccf_free_data(oriDataId))) ERR;
  if ((status = nccf_free_data(tgtDataId))) ERR;

  if ((status = nccf_free_grid(ori_grid_id))) ERR;
  if ((status = nccf_free_grid(tgt_grid_id))) ERR;

  for (i = 0; i < ndims; ++i) {
    if ((status = nccf_free_coord(ori_coord_ids[i]))) ERR;
    if ((status = nccf_free_coord(tgt_coord_ids[i]))) ERR;
  }

  return 0;
}
