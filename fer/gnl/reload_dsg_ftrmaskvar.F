	SUBROUTINE RELOAD_DSG_FTRMASKVAR (varname, slen) 

        IMPLICIT NONE
#	include	"tmap_dset.parm"
	include	'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'xdset_info.cmn_text'
	include 'xprog_state.cmn'

* 6/3/2020 *acm*
* If a variable that is a station-mask or trajectory-mask on a timeseriesProfile or
* trajectoryProfile dataset is redefined, the previous feature mask info is cleared out.  
* Here, check the variable and load the new definition as the feature mask.

	INTEGER	slen
	CHARACTER*(*) varname

	LOGICAL NC_GET_ATTRIB, got_it, reload
	INTEGER STR_SAME, iset, varid, attid, attlen, 
     .		attoutflag, maxlen, alen, status
	REAL	attval
	CHARACTER*32 attname, attstring, dummy
	CHARACTER*48 TM_FMT

* ... loop over datasets. See if the variable is listed as a feature-mask
*     for the dataset.  If so re-load the mask.

	maxlen = 32
	varid = 0

	DO iset = pdset_irrelevant+1, maxdsets
      
	   IF ( ds_name(iset) .EQ. char_init2048) CYCLE

	   varid = 0  ! look in global attributes for '__feature_mask_var'
	   attname = '__feature_mask_var'

	   CALL  CD_GET_VAR_ATT_ID (iset, varid, attname, attid, status)
	
	   IF (status .NE. ferr_ok) CYCLE
	   got_it = NC_GET_ATTRIB ( iset, varid, attname,.FALSE., dummy, 
     .                               maxlen, attlen, attoutflag, attstring,
     .                               attval )

	   IF (.NOT. got_it) CYCLE
	   IF (STR_SAME(varname(:slen), attstring(:attlen)) .EQ. 0) THEN 

* The mask has the same name as this dataset's current smask.  The LOAD
* routine also tests the length of the new mask. If it's the right 
* length for this dataset, will replace its SMASK, otherwise the 
* SMASK is canceled. 

* These attributes will get restored using the new definition for 
* the mask variable in the load routine	

	      CALL CD_DELETE_ATTRIBUTE (iset, varid, attname, status ) 
	      attname = '__feature_mask_'
	      CALL  CD_GET_VAR_ATT_ID (iset, varid, attname, attid, status)
	      CALL CD_DELETE_ATTRIBUTE (iset, varid, attname, status )  

	      cmnd_buff = 'load '//varname(:slen)

	      arg_start(1) = 6
	      arg_end(1) = 5+slen

	      reload = .TRUE.
	      CALL LOAD_DSG_MASK_FTRSET_VAR (iset, varname, reload, status)
	   ENDIF

	ENDDO

	RETURN
	END
