	SUBROUTINE IS_PROMOTE_GLOBAL( *, *, status )

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*

* create a string or float abstract array containing an attribute value
* from a global attribute of the dataset

        include 'tmap_dims.parm'
#       include "tmap_dset.parm"
        include 'xdset_info.cmn_text'
#include "netcdf.inc"
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xvariables.cmn'
	include	'xcontext.cmn'
	include	'xprog_state.cmn'

* calling argument declarations:
	INTEGER status

* internal variable declarations:
        LOGICAL	att_done
	CHARACTER*128 VAR_CODE, varname, rqst_name, attname
	CHARACTER*1   blank
	INTEGER	CGRID_SIZE, STR_SAME,
     .		mr, cx, size, n, dset, vtype, varid, attype, attlen, 
     .          ndims, nvars, ngatts, recdim, iatt, nvatts, idim, nd, 
     .          attoutflag, i, cat, attid, npts, nddims
	INTEGER*8 n8
	REAL	bad

* --- end of introductory code ---
* initialize

	cx = is_cx( isp )
	size = CGRID_SIZE(cx)
	dset = cx_data_set( cx )
	vtype = cx_type( cx )
	bad  = cx_bad_data(cx)
	blank = ' '
	attype = ptype_unknown
        
        IF (dset .LE. pdset_irrelevant) dset = cx_data_set( is_cx(isp-1) )

        CALL CD_GET_DS_INFO (dset, nddims, nvars, ngatts, recdim, status)

* reserve and catalog space for the result
* or delete and re-create if already exists?  

	mr = is_mr(isp)
	CALL CREATE_TEMP_MEM_VAR( cx, mr, status ) 
	IF ( status .NE. ferr_ok ) GOTO 5000

* diagnostic output
	IF (mode_diagnostic) 
     .      CALL DIAGNOSTIC_OUT ('global_attrib_val', mr, point_to_mr)
	
	rqst_name = VAR_CODE(cx_category(is_cx(isp-1)), cx_variable(is_cx(isp-1)))

* Is the user asking for the member dataset names?  Use attname to store the dset name

        attname =  'member_dset'
        IF (STR_SAME(attname, rqst_name) .EQ. 0) THEN 
	    CALL GET_SHORT_DSET_NAME(dset, attname, attlen)
	    nd = 1
	    CALL INIT_C_STRING_ARRAY( 1, memry(mr)%ptr,
     .				     mr_c_pointer(mr)    )
	    CALL STORE_STRING(attname(1:attlen), mr, nd-1, status)
	    GOTO 1000
        ENDIF

* Is the user asking for the value of a global attribute?

        varname = '.'
        CALL CD_GET_VAR_ID (dset, varname, varid, status)

	att_done = .FALSE.
        attloop: DO iatt = 1, ngatts
           CALL CD_GET_VAR_ATT_NAME (dset, varid, iatt, attname, status)
           IF (STR_SAME(attname, rqst_name) .EQ. 0) THEN 

              CALL CD_GET_VAR_ATT_INFO (dset, varid, iatt, 
     .                 attname, attype, attlen, attoutflag, status )

	      IF ( attype .EQ. NCCHAR ) THEN

* ... set up FORTRAN-independent pointer
	         CALL INIT_C_STRING_ARRAY( 1, memry(mr)%ptr,
     .				     mr_c_pointer(mr)    )
* ... string
	         CALL ATTRIB_STRING (cx, mr, dset, varid, 
     .			         varname, rqst_name, status)
	      ELSE

* ... array of numbers
		 IF (attlen .GT. 1) attlen = 1
		 CALL ATTRIB_VALS (memry(mr)%ptr, 
     .               size, n, dset, varid, varname, rqst_name, status)
	      ENDIF
              IF ( status .NE. ferr_ok ) GOTO 5000

	      att_done = .TRUE.
	      EXIT attloop
           ENDIF

        ENDDO attloop

* ... The attribute is not present in this member set.
*     Fill in with a blank or missing value

        IF (.NOT. att_done) THEN

           IF ( attype .EQ. ptype_unknown) THEN
              IF (vtype .EQ. ptype_string) THEN
		 attype = NCCHAR
              ELSE
		 attype = NCFLOAT
              ENDIF
           ENDIF

           IF ( attype .EQ. NCCHAR ) THEN
              CALL INIT_C_STRING_ARRAY( 1, memry(mr)%ptr,
     .				     mr_c_pointer(mr)    )
              CALL STORE_STRING(blank, mr, 0, status)
           ELSE
              CALL FILL_MEMORY ( mr, bad )
           ENDIF
        ENDIF

* result is ready to return to the stack
1000    CONTINUE
        is_mr( isp ) = mr
        mr_bad_data(mr) = cx_bad_data(cx)

	mr_protected(mr) = mr_in_progress 

* got the requested component
	RETURN 2

! RETURN 1, which is to request a sub-component, is never used ...

* error exit
 5000	RETURN
	END	

