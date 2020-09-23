
	SUBROUTINE UNSET_FEATURE_MASKS (fmask_var, slen, is_fmask, is_smask)


* 6/3/2020 *acm*
* If a variable that is a feature-mask for a DSG Dataset is redefined,
* clear out the existing feature mask info, keeping the variable attributes
* that point to the feature-mask variable.  They'll be updated when the new
* mask variable is applied.

* If they had a station-mask or trajectory-mask on a timeseriesProfile or
* trajectoryProfile dataset then we're replacing that. Flag is_smask  is true.

        IMPLICIT NONE
	include	'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'xdset_info.cmn_text'

	CHARACTER*(*) fmask_var
	LOGICAL is_fmask, is_smask
	INTEGER slen

	LOGICAL NC_GET_ATTRIB, got_it
	INTEGER STR_SAME, iset, varid, 
     .		attid, attlen, attoutflag, maxlen, status
	REAL	attval
	CHARACTER*32 attname, attstring, dummy

* initialize

	is_fmask = .FALSE.
	is_smask = .FALSE.

*... loop over datasets. See if the variable is listed as a feature-mask.
*    If redefining a feature-mask variable, keep the feature-mask attribute
*    to be used later to load the new mask on the right set(s).

	maxlen = 32

	DO iset = pdset_irrelevant+1, maxdsets
         
	   varid = 0
	   IF (dsg_msk_lm(iset) .GT. 0) THEN

	      varid = 0  ! look in global attributes for '__feature_mask_var'
	      attname = '__feature_mask_var'

	      CALL  CD_GET_VAR_ATT_ID (iset, varid, attname, attid, status)
	    
	      IF (status .NE. ferr_ok) CYCLE
	      got_it = NC_GET_ATTRIB ( iset, varid, attname, .FALSE., dummy, 
     .                               maxlen, attlen, attoutflag, attstring,
     .                               attval )

	      IF ( STR_SAME(fmask_var(:slen), attstring(:attlen)) .EQ. 0) THEN 

		 CALL FREE_LINE_DYNMEM( dsg_msk_lm(iset) )
		 CALL TM_DEALLO_DYN_LINE( dsg_msk_lm(iset) )
		 dsg_msk_lm(iset) = unspecified_int4

		 is_fmask = .TRUE.
	      ENDIF  ! this var was the mask

	   ENDIF ! mask was set on this dset

	   
	   IF (dsg_ftrsetmsk_lm(iset) .GT. 0) THEN

	      varid = 0  ! look in global attributes for '__feature_mask_var'
	      attname = '__feature_mask_var'

	      CALL  CD_GET_VAR_ATT_ID (iset, varid, attname, attid, status)
	    
	      IF (status .NE. ferr_ok) CYCLE
	      got_it = NC_GET_ATTRIB ( iset, varid, attname, .FALSE., dummy, 
     .                               maxlen, attlen, attoutflag, attstring,
     .                               attval )

	      IF ( STR_SAME(fmask_var(:slen), attstring(:attlen)) .EQ. 0) THEN 

		 CALL FREE_LINE_DYNMEM( dsg_ftrsetmsk_lm(iset) )
		 CALL TM_DEALLO_DYN_LINE( dsg_ftrsetmsk_lm(iset) )
		 dsg_ftrsetmsk_lm(iset) = unspecified_int4

		 is_smask = .TRUE.
	      ENDIF  ! this var was the mask

	   ENDIF ! mask was set on this dset

	ENDDO

	RETURN
	END
		     
		    