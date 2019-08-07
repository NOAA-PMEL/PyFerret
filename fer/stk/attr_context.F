	SUBROUTINE ATTR_CONTEXT (lo_cx, hi_cx, start, end, uvar, status)


        IMPLICIT NONE
	include 'tmap_dims.parm'
#include "netcdf.inc"
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'
        include 'xcontrol.cmn'
        include 'xalgebra.cmn'


* calling argument declarations:
	INTEGER lo_cx, hi_cx, start, end, uvar, status

* internal variable declarations:
	LOGICAL	NC_GET_ATTRIB, reset_ss, do_err, got_it, coordvar
	REAL	val
	INTEGER TM_LENSTR1, STR_SAME, 
     .		dset, dot, cat, nv, slen, maxlen, varid,
     .		ndims, nvars, ngatts, recdim, ivar, vartype, 
     .		nvdims, vdims(8), nvatts, all_outflag, attoutflag, 
     .		attlen, attype, attid, iatt
	CHARACTER name*150, attname*128, aname*128, buff*10

* set these to unspecified.  If they get set in GET_VAR_CONTEXT, 
* then do not set them to 1:attlen below.

 	cx_lo_ss(hi_cx, x_dim) = unspecified_int4
 	cx_hi_ss(hi_cx, x_dim) = unspecified_int4

	CALL GET_VAR_CONTEXT (  uvar_text(uvar)(start:end),
     .				lo_cx,
     .				hi_cx,
     .				status )
	IF ( status .NE. ferr_ok ) RETURN

* Flag to set the limits. They were not set by GET_VAR_CONTEXT, so use the
* attribute size

        reset_ss = (cx_lo_ss(hi_cx, x_dim) .EQ. unspecified_int4  .OR.
     .              cx_hi_ss(hi_cx, x_dim) .EQ. unspecified_int4 ) 

        dset = cx_data_set(hi_cx)

! See if this is a user variable and set dataset number accordingly

        dot = INDEX(uvar_text(uvar),'.')
        IF (dot .GT. 0) name = uvar_text(uvar)(1:dot-1)
        CALL FIND_VAR_NAME(pdset_irrelevant, name, cat, ivar)
        IF (  ivar .NE. munknown_var_name .AND.
     .        cat .EQ. cat_user_var) THEN
           dset = pdset_uvars

* Check whether its a LET/D variable
           CALL CD_GET_VAR_ID (dset, uvar_text(uvar)(1:dot-1), varid, status)
           IF (dset.EQ.pdset_uvars .AND. status.NE.ferr_ok) THEN
	      dset = cx_data_set(hi_cx)
	      CALL CD_GET_VAR_ID (dset, uvar_text(uvar)(1:dot-1), varid, status)
           ENDIF
           IF (status.NE.ferr_ok) do_err = .TRUE. 
        ENDIF

        IF (dset .EQ. unspecified_int4 .OR. 
     .      dset .EQ. pdset_irrelevant) THEN
           CALL TRANSFER_CONTEXT (lo_cx, hi_cx)

	   GOTO 5000
        ENDIF

* do_err: issue error message if attrib does not exist on varname.attname
        do_err = .TRUE.
        CALL  BREAK_VARATTNAME (  uvar_text(uvar)(start:end), dset, 
     .                         name, attname, varid, do_err, status )

* Is the *attr name* a pseudo-name for a variable: attnames, dimnames,
*  natts, type, ...

        IF (varid .GT. 0) THEN
           CALL CD_GET_VAR_INFO (dset, varid, name, vartype,
     .             nvdims, vdims, nvatts, coordvar, all_outflag, 
     .             status)

* attribute names
           IF (STR_SAME(attname, 'attnames') .EQ. 0 ) THEN
              cx_type(hi_cx) = ptype_string
              IF (reset_ss) THEN
                 cx_lo_ss(hi_cx, x_dim) = 1
                 cx_hi_ss(hi_cx, x_dim) = nvatts
              ELSE
                 cx_lo_ss(hi_cx, x_dim) = MAX(1, cx_lo_ss(hi_cx, x_dim))
                 cx_hi_ss(hi_cx, x_dim) = MIN(nvatts, cx_hi_ss(hi_cx, x_dim))
              ENDIF

* dimension names
           ELSE IF (STR_SAME(attname, 'dimnames') .EQ. 0 ) THEN 
              cx_type(hi_cx) = ptype_string
              IF (reset_ss) THEN
                 cx_lo_ss(hi_cx, x_dim) = 1
                 cx_hi_ss(hi_cx, x_dim) = nvdims
              ELSE
                 cx_lo_ss(hi_cx, x_dim) = MAX(1, cx_lo_ss(hi_cx, x_dim))
                 cx_hi_ss(hi_cx, x_dim) = MIN(nvdims, cx_hi_ss(hi_cx, x_dim))
              ENDIF
 
* # of dimensions for variable 
           ELSE IF (STR_SAME(attname, 'ndims') .EQ. 0 ) THEN
              cx_type(hi_cx) = ptype_float
              cx_lo_ss(hi_cx, x_dim) = 1
              cx_hi_ss(hi_cx, x_dim) = 1

*  variable types
           ELSE IF (STR_SAME(attname, 'nctype') .EQ. 0 ) THEN
              cx_type(hi_cx) = ptype_float
              cx_lo_ss(hi_cx, x_dim) = 1
              cx_hi_ss(hi_cx, x_dim) = 1

* # of attributes
            ELSE IF (STR_SAME(attname, 'nattrs') .EQ. 0 ) THEN
             cx_type(hi_cx) = ptype_float
              cx_lo_ss(hi_cx, x_dim) = 1
              cx_hi_ss(hi_cx, x_dim) = 1

            ELSE

* General attribute.
* Get attribute type and length
        
               CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
                  IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid,  
     .              attid, aname, attype, attlen, attoutflag, status )

               IF (status .NE. ferr_ok) THEN
                  GOTO 5000 ! ?? error message
               ENDIF

               IF (attype .EQ. NCCHAR) THEN
                  cx_type(hi_cx) = ptype_string
               ELSE
                  cx_type(hi_cx) = ptype_float
               ENDIF

* If the attribute we are asking for is the bad-data flag, set 
* cx_bad_data to something else, so that the value will be displayed.

               IF (STR_SAME (attname, 'missing_value') .EQ. 0 .OR.
     .            STR_SAME (attname, '_FillValue') .EQ. 0) THEN
                  maxlen = 1
                  got_it = NC_GET_ATTRIB ( dset, varid, attname, 
     .                                .TRUE., name, maxlen, attlen, 
     .                                attoutflag, buff, val)
                  IF (got_it) cx_bad_data(hi_cx) = val/2. + 1.
               ENDIF

               IF (attype .EQ. NCCHAR) THEN
                  cx_lo_ss(hi_cx, x_dim) = 1
                  cx_hi_ss(hi_cx, x_dim) = 1
               ELSE
 
* If they were not set by the var context, set the X axis limits equal 
* to the length of the array. 

                 IF (reset_ss) THEN
                     cx_lo_ss(hi_cx, x_dim) = 1
                     cx_hi_ss(hi_cx, x_dim) = attlen
                 ELSE
                    cx_lo_ss(hi_cx, x_dim) = MAX(1, cx_lo_ss(hi_cx, x_dim))
                    cx_hi_ss(hi_cx, x_dim) = MIN(attlen, cx_hi_ss(hi_cx, x_dim))
                 ENDIF
               ENDIF

            ENDIF  ! general attribute

* If the varname is ., referring to the dataset itself, then the varid = 0. These are
* dataset attributes: varnames, attnames, dimnames, ndims, nvars, ...

        ELSE !  varid = 0
           CALL CD_GET_DS_INFO (dset, ndims, nvars, ngatts, recdim, status)
     
* Count number of non-coordinate variables.
           nv = 0
           DO ivar = 1, nvars
              CALL CD_GET_VAR_INFO (dset, ivar, name, vartype, nvdims, 
     .                  vdims, nvatts, coordvar, all_outflag, status)
              IF (.NOT. coordvar) nv = nv + 1
           ENDDO

           IF (STR_SAME(attname, 'varnames') .EQ. 0 ) THEN
              cx_type(hi_cx) = ptype_string
              IF (reset_ss) THEN
                 cx_lo_ss(hi_cx, x_dim) = 1
                 cx_hi_ss(hi_cx, x_dim) = MAX(1, nv)
              ELSE
                 cx_lo_ss(hi_cx, x_dim) = MAX(1, cx_lo_ss(hi_cx, x_dim))
                 cx_hi_ss(hi_cx, x_dim) = MIN(nv, cx_hi_ss(hi_cx, x_dim))
              ENDIF

* Global attributes
           ELSE IF (STR_SAME(attname, 'attnames') .EQ. 0 ) THEN
              cx_type(hi_cx) = ptype_string
              IF (reset_ss) THEN
                 cx_lo_ss(hi_cx, x_dim) = 1
                 cx_hi_ss(hi_cx, x_dim) = MAX(1, ngatts)
              ELSE
                 cx_lo_ss(hi_cx, x_dim) = MAX(1, cx_lo_ss(hi_cx, x_dim))
                 cx_hi_ss(hi_cx, x_dim) = MIN(ngatts, cx_hi_ss(hi_cx, x_dim))
              ENDIF
                    
* # of dimensions
           ELSE IF (STR_SAME(attname, 'ndims') .EQ. 0 ) THEN
              cx_type(hi_cx) = ptype_float
              cx_lo_ss(hi_cx, x_dim) = 1
              cx_hi_ss(hi_cx, x_dim) = 1
   
* dimension names
           ELSE IF (STR_SAME(attname, 'dimnames') .EQ. 0 ) THEN
              cx_type(hi_cx) = ptype_string

              IF (reset_ss) THEN
                 cx_lo_ss(hi_cx, x_dim) = 1
                 cx_hi_ss(hi_cx, x_dim) = MAX(1, ndims)
              ELSE
                 cx_lo_ss(hi_cx, x_dim) = MAX(1, cx_lo_ss(hi_cx, x_dim))
                 cx_hi_ss(hi_cx, x_dim) = MIN(ndims, cx_hi_ss(hi_cx, x_dim))
              ENDIF
   
* coordinate-variable names
           ELSE IF (STR_SAME(attname, 'coordnames') .EQ. 0 ) THEN
              cx_type(hi_cx) = ptype_string

              IF (reset_ss) THEN
                 cx_lo_ss(hi_cx, x_dim) = 1
                 cx_hi_ss(hi_cx, x_dim) = MAX(1, nvars-nv)
              ELSE
                 cx_lo_ss(hi_cx, x_dim) = MAX(1, cx_lo_ss(hi_cx, x_dim))
                 cx_hi_ss(hi_cx, x_dim) = MIN(ndims, cx_hi_ss(hi_cx, x_dim))
              ENDIF

* # variables 
           ELSE IF (STR_SAME(attname, 'nvars') .EQ. 0 ) THEN
              cx_type(hi_cx) = ptype_float
              cx_lo_ss(hi_cx, x_dim) = 1
              cx_hi_ss(hi_cx, x_dim) = 1

* # coordinate variables (dimensions that ALSO have a coord. var)
               
           ELSE IF (STR_SAME(attname, 'ncoordvars') .EQ. 0 ) THEN
              cx_type(hi_cx) = ptype_float
              cx_lo_ss(hi_cx, x_dim) = 1
              cx_hi_ss(hi_cx, x_dim) = 1

* # of global attributes
           ELSE IF (STR_SAME(attname, 'nattrs') .EQ. 0 ) THEN
              cx_type(hi_cx) = ptype_float
              cx_lo_ss(hi_cx, x_dim) = 1
              cx_hi_ss(hi_cx, x_dim) = 1

* Is the reference to a global attribute? If so get its type and length

           ELSE
              CALL CD_GET_VAR_ID (dset, '.', varid, status)
              CALL CD_GET_DS_INFO (dset, ndims, nvars, ngatts, 
     .            recdim, status)

* If the attribute name is in quotes remove them. Perhaps the user
* is looking for a global file attribute named the same as one of our
* pseudo- attribute names 
* (Logic for case-sensitive searching will go here...)

              slen = TM_LENSTR1(attname)
              IF (attname(1:1) .EQ. "'" .AND. attname(slen:slen) .EQ. "'" ) THEN
                 attname(slen:) = " "
                 attname = attname(2:)
              ENDIF

              DO iatt = 1, ngatts
                 CALL CD_GET_VAR_ATT_NAME (dset, varid, iatt, name, status)
                 IF (STR_SAME(name, attname) .EQ. 0) THEN

                    CALL CD_GET_VAR_ATT_INFO (dset, varid, iatt, 
     .                     name, attype, attlen, attoutflag, status )

                    cx_type(hi_cx) = ptype_float
                    cx_lo_ss(hi_cx, x_dim) = 1
                    cx_hi_ss(hi_cx, x_dim) = attlen

                    IF (attype .EQ. NCCHAR) THEN
                       cx_type(hi_cx) = ptype_string
                       cx_hi_ss(hi_cx, x_dim) = 1
                    ENDIF

                 ENDIF

              ENDDO

           ENDIF  ! global att or .ncoords etc

        ENDIF  ! varid > 0 or = 0

	RETURN
 5000	RETURN
	END
