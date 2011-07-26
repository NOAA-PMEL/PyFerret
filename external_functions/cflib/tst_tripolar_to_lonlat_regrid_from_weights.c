/**
 * Test interpolation from a tripolar grid to a lon-lat grid
 *
 * tst_tripolar_to_lonlat_regrid_from_weights.c
 * Ansley Manke, from tst_tripolar_to_lonlat_regrid.c
 * Define the data field using a different function than the one used
 * in tst_tripolar_to_lonlat_regrid. Call nccf_def_regrid_from_file
 * instead of computing the regridding weights.
 *
 * $Id: tst_tripolar_to_lonlat_regrid.c 767 2011-06-06 23:20:19Z pletzer $
 *
 * \author Alexander Pletzer, Tech-X Corp.
 */

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

#include <config.h>

/* Define the values using a different function from that used in tst_tripolar_to_lonlat_regrid.c
*/
void
setDataToPrescribedValues(int nvertex, const double clon[], const double clat[], 
			  double data[]) {
  int k;
  for (k = 0; k < nvertex; ++k) {
    data[k] = sin(3.0*M_PI*clon[k]/180.0) * cos(8.0*M_PI*clat[k]/180.0);
  }
}

//////////////////////////////////////////////////////////////////////

void
setDataToMinusTwo(int nvertex, const double clon[], const double clat[],
		  double data[]) {
  int k;
  for (k = 0; k < nvertex; ++k) {
    data[k] = -2.0;
  }
}

//////////////////////////////////////////////////////////////////////

void 
createTriPolar(const int dims[], int capIndex,  
		     int coordIds[], int *gridId, int *dataId){
  const int ndims = 2;
  int nvertex = dims[0]*dims[1];
  const int save = 1;
  int status;
  double *clon, *clat;
  double *data;

  const char *dimnames[] = {"nj", "ni"};

  clat = ( double* )malloc( nvertex * sizeof( double ));
  clon = ( double* )malloc( nvertex * sizeof( double ));
  data = ( double* )malloc( nvertex * sizeof( double ));

  //if ((status = nccf_get_tripolar_grid(dims, capIndex, clon, clat))) ERR;
  double latPerim = 60.0;
  if ((status = nccf_get_bipolar_cap(dims, latPerim, -90.0, clon, clat))) ERR;

  if ((status = nccf_def_lat_coord(ndims, dims, dimnames, clat, save, &coordIds[0]))) ERR;
  if ((status = nccf_def_lon_coord(ndims, dims, dimnames, clon, save, &coordIds[1]))) ERR;
  if ((status = nccf_def_grid(coordIds, "tripolar_grid", gridId))) ERR;
  if ((status = nccf_save_grid_scrip(*gridId, "tripolar_grid_scrip.nc"))) ERR;
  if ((status = nccf_def_data(*gridId, "data", NULL, NULL, NULL, dataId))) ERR;


  /* Set the data */
  setDataToPrescribedValues(nvertex, clon, clat, data);
  if ((status = nccf_set_data_double(*dataId, data, save, 
					    NC_FILL_DOUBLE))) ERR;

  free(clat);
  free(clon);
  free(data);
}

//////////////////////////////////////////////////////////////////////

void 
createLonLat(const double xymin[], const double xymax[], 
	     const int dims[], 
	     void (*setDataFunct)(int nv, const double x[], const double y[], double d[]),
	     int coordIds[], int *gridId, int *dataId){

  const int ndims = 2;
  int nvertex = dims[0]*dims[1];
  const int save = 1;
  int status;
  int i, j, k;
  double *clon, *clat;
  double *data;
  double dxs[ndims];

  const char *dimnames[] = {"nj", "ni"};

  clat = ( double* )malloc( nvertex * sizeof( double ));
  clon = ( double* )malloc( nvertex * sizeof( double ));
  data = ( double* )malloc( nvertex * sizeof( double ));

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

  /* Set the data */
  setDataFunct(nvertex, clon, clat, data);
  if ((status = nccf_set_data_double(*dataId, data, save, 
					    NC_FILL_DOUBLE))) ERR;

  free(clat);
  free(clon);
  free(data);
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

double
checkInterp(int dataId1, int dataId2) {
  int status, k, ndims, ntot, i;
  double err;
  int gridId;
  if ((status = nccf_inq_data_gridid(dataId1, &gridId))) ERR;
  if ((status = nccf_inq_grid_ndims(gridId, &ndims))) ERR;
  int coordIds[ndims];
  if ((status = nccf_inq_grid_coordids(gridId, coordIds))) ERR;
  int dims[ndims];
  if ((status = nccf_inq_coord_dims(coordIds[0], dims))) ERR;
  ntot = 1;
  for (i = 0; i < ndims; ++i) {
    ntot *= dims[i];
  }

  /* compare the two sets of data */
  double *data1;
  double *data2;
  nc_type xtype;
  const void *fill_value;
  if ((status = nccf_get_data_pointer(dataId1, &xtype, (void **) &data1,
					     &fill_value))) ERR;
  if ((status = nccf_get_data_pointer(dataId2, &xtype, (void **) &data2,
					     &fill_value))) ERR;

  err = 0.0;
  for (k = 0; k < ntot; ++k) {
    err += fabs(data1[k] - data2[k]);
  }
  err /= ntot;

  return err;
}

//////////////////////////////////////////////////////////////////////

int main(){

  int status;
  const int ndims = 2;
  int i;
  
  int oriDims[] = {257,257}; // {17,17}; //{257, 257};
  const int capIndex = 60; //oriDims[0] / 2;
  int ori_coord_ids[ndims], ori_grid_id, oriDataId;

  int tgtDims[] = {31, 91}; //{11, 11}; //{31, 91};
  int tgt_coord_ids[ndims], tgt_grid_id, tgtDataId;
  int tgt_coord_ids_ref[ndims], tgt_grid_id_ref, tgtDataIdRef;
  double xymin[] = {60., -180.0};
  double xymax[] = {90., +180.0};

  createTriPolar(oriDims, capIndex, ori_coord_ids, &ori_grid_id, &oriDataId);
  createLonLat(xymin, xymax, tgtDims, setDataToMinusTwo,
	       tgt_coord_ids, &tgt_grid_id, &tgtDataId);
  createLonLat(xymin, xymax, tgtDims, setDataToPrescribedValues,
	       tgt_coord_ids_ref, &tgt_grid_id_ref, &tgtDataIdRef);

  writeData(ori_grid_id, oriDataId, "tst_tripolar_to_lonlat_regrid_ori_2.nc");
  writeData(tgt_grid_id_ref, tgtDataIdRef, "tst_tripolar_to_lonlat_regrid_ref_2.nc");

#ifdef HAVE_LAPACK_LIB

  /* Retrieve the weights etc from file */
  int regrid_id;
  if(( status = nccf_def_regrid_from_file( "tst_tripolar_to_lonlat_regrid_weights.nc", &regrid_id ))) ERR;

  /* Interpolate */
  if ((status = nccf_apply_regrid(regrid_id, oriDataId, tgtDataId))) ERR;

  /* Write regrided data*/
  writeData(tgt_grid_id, tgtDataId, "curv_to_lonlat_from_saved_weights_intrp.nc");

  /* Check */
  int nvalid, ntargets;
  if ((status = nccf_inq_regrid_ntargets(regrid_id, &ntargets))) ERR;
  if ((status = nccf_inq_regrid_nvalid(regrid_id, &nvalid))) ERR;
  double ratio = (double)(nvalid) / (double)(ntargets);
  double error = checkInterp(tgtDataId, tgtDataIdRef);
  printf("ratio of valid to num target points = %f  interpolation error: %f\n", 
	 ratio, error);
  //assert(ratio > 0.94);
  //assert( checkInterp(tgtDataId, tgtDataIdRef) < 0.05 );

  /* Clean up */
  if ((status = nccf_free_regrid(regrid_id))) ERR;
#endif

  if ((status = nccf_free_data(oriDataId))) ERR;
  if ((status = nccf_free_data(tgtDataId))) ERR;
  if ((status = nccf_free_data(tgtDataIdRef))) ERR;

  if ((status = nccf_free_grid(ori_grid_id))) ERR;
  if ((status = nccf_free_grid(tgt_grid_id))) ERR;
  if ((status = nccf_free_grid(tgt_grid_id_ref))) ERR;

  for (i = 0; i < ndims; ++i) {
    if ((status = nccf_free_coord(ori_coord_ids[i]))) ERR;
    if ((status = nccf_free_coord(tgt_coord_ids[i]))) ERR;
    if ((status = nccf_free_coord(tgt_coord_ids_ref[i]))) ERR;
  }

  return 0;
}
