	SUBROUTINE IS_ATTRIB_VAL(  *, *, status )

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

* create a string or float abstract array containing attribute values,
* including pseudo-names for global attributes ..nvars, ..ngatts, ..ndims, 
* ..varnames, ..gattnames, and
* varname.natts, varname.natts, varname.attnames

* return in att_cx context information describing the array of attribute values
* (based on is_constant_var.F)
* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP

* V600 8/18/05
* V600  4/06 *acm* - additional declarations found by Solaris compiler
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V694  1/15 *acm* as part of changes for ticket 2227 (handling of dimensions
*                  that are not also 1-D coordinate variables), when returning 
*                  ..dimnames, now list of all dimensions in the dataset.
* V698 *acm* 1/16 Allow for Epic CDF files, ds_type = 'ECDF'
* v698 5/16  *acm* ticket 2413: New attribute keywords ncoordvars, coordnames
*                  return info about coordinate-variables in datasets
* V710 9/16  *acm* ticket 2465: Parsing expressions including attribute references.
* 712 2/17 *sh* fix latent bug when shifting an array of attribute values
* 73  7/17 *acm fix ticket 2555, memory allocated for strings determined by the 
*               context, is not generally the full nvatts, ngatts etc

        IMPLICIT NONE
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
	INTEGER	status

* internal variable declarations:
        CHARACTER*512 varname, name
        CHARACTER*128 attname, dname, aname
        CHARACTER*128 err_string
        LOGICAL coordvar, do_err, got_it
	INTEGER	CGRID_SIZE, STR_SAME, TM_LENSTR1,
     .		mr, cx, size, uvar, n, dset, 
     .          varid, attype, attlen, ndims, nvars, 
     .          ngatts, recdim, ivar, vartype, nvdims,
     .          vdims(8), dsdims(200), iatt, nvatts, idim, nd, vd, na,
     .          attoutflag, all_outflag, xdim, i1, i2, slen,
     .          i, indx, nv, dot, cat, attid, npts, dimid, nddims, 
     .          iitem, start, end, wk_buff

	INTEGER*8 n8

* --- end of introductory code ---
* initialize

	cx = is_cx( isp )
	size = CGRID_SIZE(cx)
	dset = cx_data_set( cx )
        xdim = 1
        i1 = cx_lo_ss(cx,xdim)
        i2 = cx_hi_ss(cx,xdim)
        got_it = .FALSE.

        IF (i2 .LT. i1) GOTO 5100
        
        IF (dset .LT. pdset_uvars .OR. (.NOT.cx_dset_gvn(cx) )) 
     .      dset = cx_data_set( is_cx(isp-1) )

        CALL CD_GET_DS_INFO (dset, nddims, nvars, ngatts, 
     .               recdim, status)
        DO 100 idim = 1, nddims
	   dsdims(idim) = idim
 100	CONTINUE

* reserve and catalog space for the result
C       KMS - The ammount of memory allocated is from R_CGRID_SIZE_DELTA.
C       R_CGRID_SIZE_DELTA uses the product over idim of
C           cx_hi_ss(cx,idim - cx_lo_ss(cx,idim) / cx_delta(idim,cx)
C       (using 1 if cx_delta not specified; yes, the index order change is correct)
	CALL CREATE_TEMP_MEM_VAR( cx, mr, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* diagnostic output
	IF (mode_diagnostic) 
     .      CALL DIAGNOSTIC_OUT ('attrib_val',mr,point_to_mr)

* parse and insert the constant array

	uvar = cx_variable(cx)

! See if this is a user variable and set dataset number accordingly

        dot = INDEX(uvar_text(uvar),'.')
        IF (dot .GT. 0) varname = uvar_text(uvar)(1:dot-1)
        CALL FIND_VAR_NAME(pdset_irrelevant, varname, cat, ivar)
        IF (  ivar .NE. munknown_var_name .AND.
     .         cat .EQ. cat_user_var) dset = pdset_uvars

* do_err: issue error message if attrib does not exist on varname.attname
        do_err = .TRUE. 
	
	iitem = 0
	DO i = 1, uvar_num_items(uvar)
 	   IF (dot .GE. uvar_item_start(i,uvar) .AND. 
     .         dot .LT. uvar_item_end(i,uvar)) iitem = i
	ENDDO
	
	start = 1
	end = TM_LENSTR1(uvar_text(uvar))
	IF (iitem .GT. 0) THEN
	   start = uvar_item_start(iitem,uvar)
	   end = uvar_item_end(iitem,uvar)
	ENDIF
        CALL BREAK_VARATTNAME ( uvar_text(uvar)(start:end), dset, varname, 
     .                          attname, varid, do_err, status )
 
        IF (STR_SAME (varname, '.') .EQ. 0) THEN

           CALL CD_GET_DS_INFO (dset, ndims, nvars, ngatts, 
     .               recdim, status)

           IF (STR_SAME(attname, 'varnames') .EQ. 0 ) THEN

* Get and store the variable names.

              n = i2 - i1 + 1
              CALL INIT_C_STRING_ARRAY( n,
     .				     memry(mr)%ptr,
     .				     mr_c_pointer(mr)    )
              n = 0
              nv = 0
              DO 120 i = 1, nvars
                 name = ' '
                 CALL CD_GET_VAR_INFO (dset, i, name, vartype,
     .                nvdims, vdims, nvatts, coordvar, all_outflag, 
     .                status)
                 IF (.NOT. coordvar) THEN
                    nv = nv + 1
                    IF (nv .GE. i1 .AND. nv .LE. i2) THEN
                       slen = TM_LENSTR1(name)
                       CALL STORE_STRING(name(1:slen), mr, n, status)
                       n = n + 1
                    ENDIF
                 ENDIF
 120          CONTINUE
              got_it = .TRUE.
	      
	     
* dimension names that are also coordinate variables
           ELSE IF (STR_SAME(attname, 'coordnames') .EQ. 0 ) THEN

              n = i2 - i1 + 1
              CALL INIT_C_STRING_ARRAY( n,
     .				     memry(mr)%ptr,
     .				     mr_c_pointer(mr)    )
              n = 0
              nv = 0
              DO 130 i = 1, nvars
                 name = ' '
                 CALL CD_GET_VAR_INFO (dset, i, name, vartype,
     .                nvdims, vdims, nvatts, coordvar, all_outflag, 
     .                status)
                 IF (coordvar) THEN
                    nv = nv + 1
                    IF (nv .GE. i1 .AND. nv .LE. i2) THEN
                       slen = TM_LENSTR1(name)
                       CALL STORE_STRING(name(1:slen), mr, n, status)
                       n = n + 1
                    ENDIF
                 ENDIF
 130          CONTINUE
              got_it = .TRUE.
 
           ELSE IF (STR_SAME(attname, 'dimnames') .EQ. 0 ) THEN
	   

* Get and store the dimension names.

              n = i2 - i1 + 1
	      CALL INIT_C_STRING_ARRAY( n,
     .				     memry(mr)%ptr,
     .				     mr_c_pointer(mr)    )
                IF (ds_type(dset).NE.'CDF' .AND. ds_type(dset).NE.'ECDF') GOTO 5200

* All the dimensions from the dataset

              n = 0
              nv = 0
              nvdims = nddims
	      
              DO 220 idim = 1, nvdims
                 name = ' '
		 CALL CD_GET_DS_DIMS (dset, dsdims(idim), name, npts, status)
		 nv = nv + 1
		 IF (nv .GE. i1 .AND. nv .LE. i2) THEN
                    slen = TM_LENSTR1(name)
                    CALL STORE_STRING(name(1:slen), mr, n, status)
                    n = n + 1
		 ENDIF
 220          CONTINUE
              got_it = .TRUE.
 
           ELSE IF (STR_SAME(attname, 'attnames') .EQ. 0 ) THEN
              IF (ngatts .GT. 0) THEN
                 CALL CD_GET_VAR_ID (dset, '.', varid, status)

	         n = i2-i1+1
		 CALL INIT_C_STRING_ARRAY( n,
     .				     memry(mr)%ptr,
     .				     mr_c_pointer(mr)    )
                 n = 0
                 DO 300 iatt = i1, i2
                 CALL CD_GET_VAR_ATT_NAME (dset, varid, iatt, name, 
     .                    status)
                 slen = TM_LENSTR1(name)
                 CALL STORE_STRING(name(1:slen), mr, n, status)
                 n = n + 1
 300          CONTINUE
              ENDIF
              got_it = .TRUE.

           ELSE IF (STR_SAME(attname, 'ndims') .EQ. 0 ) THEN
              IF (ds_type(dset).NE.'CDF' .AND. ds_type(dset).NE.'ECDF') GOTO 5200

              memry(mr)%ptr(1) = ndims
              got_it = .TRUE.

           ELSE IF (STR_SAME(attname, 'nattrs') .EQ. 0 ) THEN
              memry(mr)%ptr(1) = ngatts
              got_it = .TRUE.

           ELSE IF (STR_SAME(attname, 'nvars') .EQ. 0 .OR.
     .              STR_SAME(attname, 'ncoordvars') .EQ. 0) THEN

* Count number of non-coordinate variables.
              n = 0
              DO 400 ivar = 1, nvars
                 CALL CD_GET_VAR_INFO (dset, ivar, name, vartype,
     .                nvdims, vdims, nvatts, coordvar, all_outflag, 
     .                status)
                 IF (.NOT. coordvar) n = n + 1
 400          CONTINUE
              IF (STR_SAME(attname, 'ncoordvars') .EQ. 0) n = nvars-n
              memry(mr)%ptr(1) = n
              got_it = .TRUE.
           ENDIF

* Is the user asking for the value of a global attribute?

           IF (got_it) GOTO 1000

           CALL CD_GET_VAR_ID (dset, '.', varid, status)

           DO 500 iatt = 1, ngatts
              CALL CD_GET_VAR_ATT_NAME (dset, varid, iatt, name, 
     .                    status)
              IF (STR_SAME(name, attname) .EQ. 0) THEN 

                 CALL CD_GET_VAR_ATT_INFO (dset, varid, iatt, 
     .                 name, attype, attlen, attoutflag, status )

	         IF ( attype .EQ. NCCHAR ) THEN

* ... set up FORTRAN-independent pointer

	            CALL INIT_C_STRING_ARRAY( 1,
     .				     memry(mr)%ptr,
     .				     mr_c_pointer(mr)    )

* ... string
	            CALL ATTRIB_STRING (cx, mr, dset, varid, 
     .			         varname, attname, status)
	         ELSE
* ... array of numbers
	            CALL ATTRIB_VALS (memry(mr)%ptr, 
     .                size, n, dset, varid, varname, attname, status)
	         ENDIF
                 IF ( status .NE. ferr_ok ) GOTO 5000
              ELSE

* If the attribute name is in quotes try to get it anyway, as a numeric or
* string attribute. Perhaps the user is looking for a file attribute named 
* the same as one of our pseudo-attribute names 
* (Logic for case-sensitive searching will go here...)

                 slen = TM_LENSTR1(attname)
                 IF (attname(1:1) .EQ. "'" .AND. 
     .               attname(slen:slen) .EQ. "'" ) THEN

	            CALL ATTRIB_VALS (memry(mr)%ptr, 
     .                 size, n, dset, varid, varname, attname, status)
                    IF (status .NE. ferr_ok ) CALL ATTRIB_STRING (cx, mr, 
     .	                      dset, varid,  varname, attname, status)
                 ENDIF

              ENDIF
 500          CONTINUE

        ELSE  ! variable name is not '.'

           CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid,
     .               status)
           IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .        attid, aname, attype, attlen, attoutflag, status )

           IF (status .NE. ferr_ok) THEN

* See if it is a pseudo-attribute attnames, ndims, etc.
              CALL CD_GET_VAR_INFO (dset, varid, name, vartype,
     .                nvdims, vdims, nvatts, coordvar, all_outflag, 
     .                status)

              IF (STR_SAME(attname, 'attnames') .EQ. 0 ) THEN
                 IF (nvatts .GT. 0) THEN

	           n = i2-i1+1
		   CALL INIT_C_STRING_ARRAY( n,
     .				     memry(mr)%ptr,
     .				     mr_c_pointer(mr)    )
                   n = 0
                   DO 600 iatt = i1, i2
                      CALL CD_GET_VAR_ATT_NAME (dset, varid, iatt, name, 
     .                    status)
                      slen = TM_LENSTR1(name)
                      CALL STORE_STRING(name(1:slen), mr, n, status)
                      n = n + 1
 600               CONTINUE
                ENDIF
                got_it = .TRUE.

             ELSE IF (STR_SAME(attname, 'dimnames') .EQ. 0 ) THEN

* Get the datset dimension names - not all dimensions are a coordinate variable.

                IF (nvdims .GT. 0) THEN
		
	           n = i2-i1+1
	           CALL INIT_C_STRING_ARRAY( n,
     .				     memry(mr)%ptr,
     .				     mr_c_pointer(mr)    )
                IF (ds_type(dset).NE.'CDF' .AND. ds_type(dset).NE.'ECDF') GOTO 5200

                   n = 0
                   DO 700 idim = i1, i2
		      CALL CD_GET_DS_DIMS (dset, dsdims(vdims(idim)), 
     .				     dname, npts, status)
                      slen = TM_LENSTR1(dname)
                      CALL STORE_STRING(dname(1:slen), mr, n, status)
                      n = n + 1
 700               CONTINUE
                ENDIF
                got_it = .TRUE.

             ELSE IF (STR_SAME(attname, 'ndims') .EQ. 0 ) THEN
                IF (ds_type(dset).NE.'CDF' .AND. ds_type(dset).NE.'ECDF') GOTO 5200

                memry(mr)%ptr(1) = nvdims
                got_it = .TRUE.

             ELSE IF (STR_SAME(attname, 'nctype') .EQ. 0 ) THEN

                memry(mr)%ptr(1) = vartype
                got_it = .TRUE.

             ELSE IF (STR_SAME(attname, 'nattrs') .EQ. 0 ) THEN
                memry(mr)%ptr(1) = nvatts
                got_it = .TRUE.

             ENDIF
             IF (got_it) GOTO 1000

* If the attribute name is in quotes remove them. Perhaps the user
* is looking for a file attribute named the same as one of our pseudo-
* attribute names 
* (Logic for case-sensitive searching will go here...)

             slen = TM_LENSTR1(attname)
             IF (attname(1:1) .EQ. "'" .AND. attname(slen:slen) .EQ. "'" ) THEN
                attname = attname(2:slen-1)
                attname(slen-1:slen) = ""
             ENDIF

             CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid,
     .               status)
             IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .          attid, aname, attype, attlen, attoutflag, status )

             IF (status .EQ. ferr_ok) GOTO 1000

           ELSE

* Is it a regular attribute from the file or a definition

 900	      IF ( attype .EQ. NCCHAR ) THEN
* ... set up FORTRAN-independent pointer
	         CALL INIT_C_STRING_ARRAY( 1,
     .				     memry(mr)%ptr,
     .				     mr_c_pointer(mr)    )

* ... string
	         CALL ATTRIB_STRING (cx, mr, dset, varid, varname, attname,
     .			          status)
	      ELSE
* ... array of numbers

* 2/17 - for a case like 
*	define att v.att = {1, 2, 3, 4, 5}
*	load v.att[i=3:5]
* the allocated memory will not be large enough to accommodate the full array
* of attribute values, so we need to allocate a work array
! since other calls are ignoring return status, we will, too ...
	         n8 = attlen
	         CALL CREATE_DYN_WRKSPC( n8, wk_buff, status )
	         CALL ATTRIB_VALS (workmem(wk_buff)%ptr, size, n, dset, 
     .                varid, varname, attname, status)
     
* ... left-shift the results to start a context low limit
                 indx = 0
                 DO 800 i = i1, i2
                    indx = indx + 1
                    memry(mr)%ptr(indx) = workmem(wk_buff)%ptr(i)
800              CONTINUE

	         CALL FREE_WS_DYNMEM( wk_buff )

	      ENDIF
	      IF ( status .NE. ferr_ok ) GOTO 5000
           ENDIF

        ENDIF
* result is ready to return to the stack
1000    CONTINUE
        is_mr( isp ) = mr
        mr_bad_data(mr) = cx_bad_data(cx)

* got the requested component
	RETURN 2

! RETURN 1, which is to request a sub-component, is never used ...

* error exit
 5000	RETURN
 5100   status = ferr_out_of_range
        err_string = "invalid dataset attribute index"
        CALL ERRMSG( status, status, err_string, *5000 )
 5200   status = ferr_invalid_command
        err_string = "implemented only for netCDF/OPeNDAP datasets"
        CALL ERRMSG( status, status, err_string, *5000 )
	END	

