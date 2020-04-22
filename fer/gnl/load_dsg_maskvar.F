	SUBROUTINE LOAD_DSG_MASKVAR (dset, fmask_var, status)

c set data/fmask=

* If a feaure-mask was given, store the values of the mask in line storage. 
* Save the memory location dsg_msk_lm(dset), in common. Mask data used in creating 
* feature-masks when computing feature-level masking which also may include 
* context limits in the E direftion. Will deallocate line mem dsg_msk_lm on 
* defining a new mask, closing the dataset, opr CANCEL DATA/FMASK
* 
* 4/20/2020 *acm* update to error message

	include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'slash.parm'
	include 'command.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xprog_state.cmn'
#include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	include 'xtm_grid.cmn_text'
	include 'netcdf.inc'

	INTEGER dset, status
	CHARACTER*(*) fmask_var
	
	LOGICAL TM_LEGAL_NAME, TM_ITSA_DSG
	INTEGER TM_DSG_DSET_FROM_GRID, TM_DSG_NFEATURES, 
     .		TM_LENSTR1, CX_DIM_LEN,
     .		i, ivar, grid, loc,  ndim, idir, npts, cx, mr1, 
     .		nfeatures, idim, dim(nferdims), attlen, varid, 
     .		attid, attype, attoutflag
	CHARACTER FULL_VAR_TITLE*128, TM_FMT*12,  
     .		buff*128, mask_title*128
	REAL	dummy

c Make sure the name is ok
	
	   IF (fmask_var .EQ. ' ') GOTO 5100
	   IF ( .NOT.TM_LEGAL_NAME(fmask_var) ) GOTO 5200


c find a variable in this dataset and test whether it's a dsg dataset
	      
	DO ivar = 1, maxvars
	   IF ( ds_var_setnum(ivar) .EQ. dset ) THEN
              grid = ds_grid_number(ivar)
              EXIT
	   ENDIF
	ENDDO

	IF (.NOT. TM_ITSA_DSG( grid ) ) GOTO 5300

	IF (dset .NE. TM_DSG_DSET_FROM_GRID( grid ) ) GOTO 5300
	nfeatures = TM_DSG_NFEATURES( grid )	


* Set the line-memory to store the mask data

* ... A mask was already defined. Wipe that one out and we'll redefine it.

	IF (dsg_msk_lm(dset) .NE. unspecified_int4)  THEN
	   CALL FREE_LINE_DYNMEM( dsg_msk_lm(dset) )
	   CALL TM_DEALLO_DYN_LINE( dsg_msk_lm(dset) )
	   dsg_msk_lm(dset) = unspecified_int4

* ... wipe memory clear of stored variables - this could change all definitions
	   DO i = 1,max_mr_avail
	      IF ( mr_protected( i ) .NE. mr_deleted ) 
     .               CALL DELETE_VARIABLE( i )
	   ENDDO

	ENDIF

* Load the variable. Reset the arg pointers to get the mask-var           
	
	loc = qual_given(slash_set_data_fmask)
	num_args = 1	 
	arg_end(1) = qual_end(loc)
	arg_start(1) = qual_end(loc) - TM_LENSTR1(fmask_var) + 1

	CALL GET_CMND_DATA ( cx_last, ptype_float, status )
	IF (status .NE. ferr_ok .OR. num_uvars_in_cmnd .GT. 1) GOTO 5400

* ... make sure it's a line (not a point,plane,etc.) and that the M extent
*     is the feature-length

	cx = is_cx(isp)
	CALL GET_CX_DIMS( cx, ndim, dim )
	IF ( ndim .GT. 1 ) GOTO 5400
	idim = dim(1)
	IF (nfeatures .NE. CX_DIM_LEN( idim, cx ) ) GOTO 5400


* store it here,

	CALL TM_ALLO_TMP_LINE(dsg_msk_lm(dset), status)
	CALL GET_LINE_DYNMEM (nfeatures, dsg_msk_lm(dset), status)
	CALL TM_USE_LINE(dsg_msk_lm(dset))

	CALL TM_NEW_LINE_NAME ('FEATURE_MASK', buff ) 
	line_name(dsg_msk_lm(dset)) = buff
	line_direction( dsg_msk_lm(dset) ) = 'EE'

	mr1 = is_mr( num_uvars_in_cmnd )
	cx = is_cx( num_uvars_in_cmnd )

	CALL EXTRACT_LINE (cx,
     .			memry(mr1)%ptr,
     .			mr1,
     .			linemem(dsg_msk_lm(dset))%ptr,
     .			idir,
     .			ndim,
     .			npts,
     .			status	)

	IF (npts.NE.nfeatures .OR. status.NE.ferr_ok) GOTO 5000

* Put the default bad-flag into the mask data
	CALL TM_SWITCH_BAD ( cx_bad_data (cx), bad_val4, linemem(dsg_msk_lm(dset))%ptr, npts )

* Set a global attribute with the mask title

	buff = '__feature_mask_'
	attlen = TM_LENSTR1(buff)
	mask_title = FULL_VAR_TITLE( cx, .FALSE., attlen )

	varid = 0
	CALL CD_GET_VAR_ATT_ID (dset, varid, buff, attid, status) 
	IF (attid .GT. 0) THEN
	   CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, 
     .                buff, attype, attlen, attoutflag, status )
	   attoutflag = 0
	   CALL CD_REPLACE_ATTR (dset, varid, buff, attype, attlen, 
     .              mask_title, dummy, status)
	ELSE
	   attype = NCCHAR
	   attoutflag = 0
	   CALL CD_PUT_NEW_ATTR (dset, varid, buff, attype, 
     .                 attlen, attoutflag, mask_title, dummy, status )
	ENDIF
	IF (status .NE. ferr_ok) GOTO 5000

	status = ferr_ok

 5000	RETURN

 5100	CALL ERRMSG( ferr_unknown_arg, status, 'argument required /FMASK=?', *5000)
 5200	CALL ERRMSG( ferr_invalid_command, status,
     .			'/FMASK=name is not an acceptable name', *5000)
 5300	CALL ERRMSG (ferr_invalid_command, status,
     .'/FMASK is set only for Dicrete Sampling Geometries datasets',
     .               						*5000 )
 5400	buff = TM_FMT(DBLE(nfeatures), 0, 12, i) 
	CALL ERRMSG (ferr_invalid_command, status,
     .'Mask variable must be 1-dimensional, with length num-features = '//
     .   buff(:i),						*5000 )
	END
