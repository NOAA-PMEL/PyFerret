
	SUBROUTINE CD_DSG_TP_TO_PROF (dset)

* !!!!
* For timeseriesProfile datasets, the lon and lat coordinates are given in the 
* timeseries-station index direction. Use the station index variable to redefine the 
* dsg-coordiate variables for the translation grid to list these coordinates in the 
* profile dimension


* include files
        include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
        include 'xbuild_grids.cmn'
        include 'xdset_info.cmn_text'
        external xdset_info_data
        include 'xdyn_linemem.cmn_text'
        include 'xtm_grid.cmn_text'
        include 'xunits.cmn_text'

	INTEGER dset

	INTEGER idim, ista, iprof, nstations, nprofiles, lm_coord, lm_index
	REAL	sta_index, coord

	REAL, DIMENSION(:), ALLOCATABLE :: coord_data
	REAL, DIMENSION(:), ALLOCATABLE :: station_index

	nprofiles = line_dim (dsg_instance_axis) ! e=dimension size
	nstations = line_dim (dsg_instnc_ftrset_axis) ! f=dimension size 

	ALLOCATE (coord_data(nstations))
	ALLOCATE (station_index(nprofiles))

* Get the station-index number for the profiles.
* These are zero-based, so add 1 for comparing to profile-number below.

	lm_index = dsg_loaded_lm(dsg_index_var(dset))

	DO ista = 1, nprofiles
	   station_index(ista) = dsg_linemem(lm_index)%ptr(ista) + 1
	ENDDO

* Get coordinates, longitude then latitude. Store them, per-profile

	DO idim = x_dim, y_dim

* Get lon or lat per station
	   lm_coord = dsg_loaded_lm(dsg_coord_var(idim,dset))
	   DO iprof = 1, nstations
	      coord_data(iprof) = dsg_linemem(lm_coord)%ptr(iprof)
	   ENDDO

* which profile number is at station sta_index

	   DO iprof = 1, nprofiles
	      ista = station_index(iprof)
	      CALL PUT_LINE_COORD ( dsg_linemem(lm_coord)%ptr, 
     .		      iprof, coord_data(ista) ) 
	   ENDDO

	ENDDO

	DEALLOCATE(coord_data)
	DEALLOCATE(station_index)

	RETURN
	END
