c*
c* ef_get_its_dsg.F
c*
c* Ansley Manke
c*  May 2020
c*
c*
c*
c* This returns a flag, its_dsg = true if the datset is
c* a discrete-sampling-geometries (DSG) datset.  If it's DSG
c* return the feature type and number of features.

	SUBROUTINE EF_GET_ITS_DSG(id, iarg, its_dsg )

	IMPLICIT NONE
	INCLUDE 'tmap_dims.parm'
	INCLUDE 'ferret.parm'
	INCLUDE 'xvariables.cmn'
	INCLUDE 'xcontext.cmn'
	include 'xtm_grid.cmn_text'
	include 'xdset_info.cmn_text'
	INCLUDE 'EF_Util.parm'

	INTEGER id, iarg
	LOGICAL its_dsg 
	
	INTEGER TM_DSG_NFEATURES, 
     .		grid, cx_list(EF_MAX_ARGS)
	LOGICAL TM_ITSA_DSG
 
c* The context list was cached by Ferret and must be retrieved
c* to be accessible to the external function. 

	CALL EF_GET_CX_LIST(cx_list)
	
	grid = cx_grid( cx_list(iarg) )
	its_dsg = TM_ITSA_DSG( grid ) 

* can call other functions such as TM_DSG_NFEATURES or TM_DSG_FACTS
* when working to optimize functions for dsg data types

	RETURN 
	END
