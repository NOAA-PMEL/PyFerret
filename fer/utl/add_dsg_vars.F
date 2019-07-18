	SUBROUTINE ADD_DSG_VARS (dset, cbuff)
	
* 2/2019  Ansley Manke
* NOAA/PMEL, Seattle, WA - Science Data Integration Group

* Add variables to the command-line buffer for the feature-id variable
* Then they're read and ready for CDF_LIST_DSG to write them out.
* feature-variable is often of type string and is not stored at the
* time of dataset initialization.

        include 'tmap_dims.parm'
        include 'xtm_grid.cmn_text'
	include 'ferret.parm'
	include 'xdset_info.cmn_text'

	INTEGER dset
	CHARACTER*(*) cbuff

	LOGICAL relevant_coord(4)
	INTEGER TM_LENSTR1, idim, gxlate, line, status
	CHARACTER*128 SANITARY_VAR_CODE, vname
	
* Probably just need the feature-id variable

	vname = SANITARY_VAR_CODE(cat_file_var, dsg_feature_var(dset) )
	cbuff = cbuff(:TM_LENSTR1(cbuff))//','//vname(:TM_LENSTR1(vname))
	   
	RETURN
	END
