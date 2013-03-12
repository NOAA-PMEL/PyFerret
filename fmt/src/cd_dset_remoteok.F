
	  SUBROUTINE CD_DSET_REMOTEOK(dset, ok_remote)

C Check whether we can define a LET/REMOTE variable in this dataset.
* Sets the flag  ds_accepts_remote for the dataset.

* include files
#include "netcdf.inc"
	include 'tmap_errors.parm'
#include "gt_lib.parm"
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xdset_info.cd_equiv'

	INTEGER dset
	LOGICAL ok_remote

	LOGICAL TM_HAS_STRING
	INTEGER TM_LENSTR1, slen, vlen, ivar, cdfid, cdfstat
	CHARACTER vname*128, buff*3000, ebuff*3000

* Already checked? 
	IF (ds_accepts_remote(dset)) GOTO 5000

* If not an opendap set, then it does not accept remote definitions
	IF ( ds_des_name(dset)(1:7) .NE. 'http://') GOTO 5100

	DO 100 ivar = 1, maxvars
	   IF (ds_var_setnum(ivar) .EQ. dset) GOTO 110
 100	CONTINUE
        GOTO 5000	 !no variables in the datset, so return FALSE
 110	CONTINUE

	slen = TM_LENSTR1(ds_des_name(dset))

* If its already a datset with f-tds variable in it, it cannot be used.
* (depending on where we're calling this, there should be a message).

	IF (TM_HAS_STRING(ds_des_name(dset)(1:slen), 'letdeq1')) GOTO 5000

* Name of first variable, use it to define a variable name.

	vname = ds_var_code(ivar)
	vlen = TM_LENSTR1(vname)

* Construct a url which is the original url, with a variable defined,
* LET varname_1_NEW = 7
	
	buff = '{}{letdeq1 '//vname(:vlen)//'1_new=7}'
	slen = TM_LENSTR1(buff)

* Encode the variable definition, set up the full url, 
* and try to open it.

	CALL CD_ENCODE_URL ( buff, ebuff, status)

	slen = TM_LENSTR1(ds_des_name(dset))
	vlen = TM_LENSTR1(ebuff)
	buff = ds_des_name(dset)(:slen)//'_expr_'//ebuff(:vlen)

#ifdef usingDODSf2cUnderscore 
	cdfstat = NF_OPEN_(buff, NF_NOWRITE, cdfid) 
#else
	cdfstat = NF_OPEN(buff, NF_NOWRITE, cdfid) 
#endif

* If it could be opened, we can make remote-variable definitions.

	IF (cdfstat .EQ. NF_NOERR) THEN
	   ds_accepts_remote(dset) = .TRUE.

* close the url with f-tds var - we were just testing here.
#ifdef usingDODSf2cUnderscore
	   cdfstat = NF_CLOSE_( cdfid )
#else
	   cdfstat = NF_CLOSE( cdfid )
#endif

	ELSE
	   ds_accepts_remote(dset) = .FALSE.
	   CALL WARN ('dataset does not accept REMOTE variable definitions')
	ENDIF

 5000	ok_remote = ds_accepts_remote(dset)
	RETURN

 5100	CONTINUE
	ds_accepts_remote(dset) = .FALSE.
	CALL WARN ('dataset does not accept REMOTE variable definitions')
	END

