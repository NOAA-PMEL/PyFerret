	SUBROUTINE GCF_GET_ALT_FCN_DIR_ARG( gcfcn, idir_args )

* return the argument that sets a direction indicating 
* an alternative function.


        IMPLICIT NONE
	include	'tmap_dims.parm'
	include 'ferret.parm'
	include 'grid_chg_fcns.parm' 
	include 'xgrid_chg_fcns.cmn'

* calling argument declarations
	INTEGER		gcfcn, idir_args(*)

	INTEGER idim

	DO idim = 1, nferdims
	   idir_args(idim) = gfcn_dir_args (idim, gcfcn)
	ENDDO

	RETURN
	END
