	SUBROUTINE IS_ATTRIB_VAL( memory, *, *, status )

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

        include 'netcdf.decl'
#include "netcdf.inc"
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xvariables.cmn'
	include	'xcontext.cmn'
	include	'xprog_state.cmn'

* calling argument declarations:
	INTEGER	status
	REAL	memory( mem_blk_size, max_mem_blks )

* internal variable declarations:
        CHARACTER*512 varname, name
        CHARACTER*128 attname, dname
        LOGICAL coordvar, do_err
	INTEGER	CGRID_SIZE, STR_SAME, TM_LENSTR1,
     .		mr, cx, size, uvar, n, dset, 
     .          varid, attype, attlen, ndims, nvars, 
     .          ngatts, recdim, ivar, vartype, nvdims,
     .          vdims(8), iatt, nvatts, idim, nd, vd, na,
     .          attoutflag, all_outflag, xdim, i1, i2, slen,
     .          i, indx, nv

* --- end of introductory code ---
* initialize

	cx = is_cx( isp )
	size = CGRID_SIZE(cx)
	dset = cx_data_set( cx )
        xdim = 1
        i1 = cx_lo_ss(cx,xdim)
        i2 = cx_hi_ss(cx,xdim)
        
        IF (dset .LE. 0 .OR. (.NOT.cx_dset_gvn(cx) )) 
     .      dset = cx_data_set( is_cx(isp-1) )

* reserve and catalog space for the result

	CALL CREATE_TEMP_MEM_VAR( cx, mr, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* diagnostic output
	IF (mode_diagnostic) 
     .      CALL DIAGNOSTIC_OUT ('attrib_val',mr,point_to_mr)

* parse and insert the constant array

	uvar = cx_variable(cx)

* do_err: issue error message if attrib does not exist on varname.attname
        do_err = .TRUE. 
        CALL BREAK_VARATTNAME ( uvar_text(uvar), dset, varname, 
     .                          attname, varid, do_err, status )
 
        IF (STR_SAME (varname, '.') .EQ. 0) THEN

           CALL CD_GET_DS_INFO (dset, ndims, nvars, ngatts, 
     .               recdim, status)

           IF (STR_SAME(attname(:4), 'varn') .EQ. 0 ) THEN  ! varnames

* Get and store the variable names.

              n = i2 - i1 + 1
              CALL INIT_C_STRING_ARRAY( n,
     .				     memory(1,mr_blk1(mr)),
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
 
           ELSE IF (STR_SAME(attname(:4), 'dimn') .EQ. 0 ) THEN  ! dimnames

* Get and store the coordinate variable names.

              n = i2 - i1 + 1
	      CALL INIT_C_STRING_ARRAY( n,
     .				     memory(1,mr_blk1(mr)),
     .				     mr_c_pointer(mr)    )
              n = 0
              nv = 0
              DO 220 i = 1, nvars  
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
 220          CONTINUE                

           ELSE IF (STR_SAME(attname(:3), 'att') .EQ. 0 ) THEN  ! attnames
              IF (ngatts .GT. 0) THEN
                 CALL CD_GET_VAR_ID (dset, '.', varid, status)

	         CALL INIT_C_STRING_ARRAY( ngatts,
     .				     memory(1,mr_blk1(mr)),
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

           ELSE IF (STR_SAME(attname(:4), 'ndim') .EQ. 0 ) THEN  ! ndims
              memory( 1, mr_blk1(mr)) = ndims

           ELSE IF (STR_SAME(attname(:3), 'nat') .EQ. 0 ) THEN   ! natts
              memory( 1, mr_blk1(mr)) = ngatts

           ELSE IF (STR_SAME(attname(:4), 'nvar') .EQ. 0 ) THEN  ! nvars

* Count number of non-coordinate variables.
              n = 0
              DO 400 ivar = 1, nvars
                 CALL CD_GET_VAR_INFO (dset, ivar, name, vartype,
     .                nvdims, vdims, nvatts, coordvar, all_outflag, 
     .                status)
                 IF (.NOT. coordvar) n = n + 1
 400          CONTINUE
              memory( 1, mr_blk1(mr)) = n

           ENDIF

* Is the user asking for the value of a global attribute?

           CALL CD_GET_VAR_ID (dset, '.', varid, status)
           DO 500 iatt = 1, ngatts
              CALL CD_GET_VAR_ATT_NAME (dset, varid, iatt, name, 
     .                    status)
              IF (STR_SAME(name, attname) .EQ. 0) THEN 

                 CALL CD_GET_VAR_ATT_INFO (dset, varid, attname, 
     .             attype, attlen, attoutflag, status)

	         IF ( attype .EQ. NCCHAR ) THEN

* ... set up FORTRAN-independent pointer

	            CALL INIT_C_STRING_ARRAY( 1,
     .				     memory(1,mr_blk1(mr)),
     .				     mr_c_pointer(mr)    )

* ... string
	            CALL ATTRIB_STRING (cx, mr, dset, varid, 
     .			         varname, attname, status)
	         ELSE
* ... array of numbers
	            CALL ATTRIB_VALS (memory( 1, mr_blk1(mr)), 
     .                size, n, dset, varid, varname, attname, status)
	         ENDIF
                 IF ( status .NE. ferr_ok ) GOTO 5000
              ENDIF
 500          CONTINUE

        ELSE

           CALL CD_GET_VAR_ATT_INFO (dset, varid, attname, attype, attlen, 
     .          attoutflag, status)

           IF (status .NE. ferr_ok) THEN

* See if it is a pseudo-attribute attnames, ndims, etc.
              CALL CD_GET_VAR_INFO (dset, varid, name, vartype,
     .                nvdims, vdims, nvatts, coordvar, all_outflag, 
     .                status)

             IF (STR_SAME(attname(:4), 'attn') .EQ. 0 ) THEN  ! attnames
              IF (nvatts .GT. 0) THEN

	         CALL INIT_C_STRING_ARRAY( nvatts,
     .				     memory(1,mr_blk1(mr)),
     .				     mr_c_pointer(mr)    )
                 n = 0
                 DO 600 iatt = i1, i2
                 CALL CD_GET_VAR_ATT_NAME (dset, varid, iatt, name, 
     .                    status)
                 slen = TM_LENSTR1(name)
                 CALL STORE_STRING(name(1:slen), mr, n, status)
                 n = n + 1
 600          CONTINUE
              ENDIF

             ELSE IF (STR_SAME(attname(:4), 'dimn') .EQ. 0 ) THEN  ! dimnames
              IF (nvdims .GT. 0) THEN

	         CALL INIT_C_STRING_ARRAY( nvdims,
     .				     memory(1,mr_blk1(mr)),
     .				     mr_c_pointer(mr)    )
                 n = 0
                 DO 700 idim = i1, i2
                    CALL CD_GET_VAR_INFO (dset, vdims(idim), dname, 
     .                 vartype, nd, vd, na, coordvar, all_outflag, 
     .                 status)
                    slen = TM_LENSTR1(dname)
                    CALL STORE_STRING(dname(1:slen), mr, n, status)
                    n = n + 1
 700             CONTINUE
              ENDIF

             ELSE IF (STR_SAME(attname(:4), 'ndim') .EQ. 0 ) THEN  ! ndims
                memory( 1, mr_blk1(mr)) = nvdims

             ELSE IF (STR_SAME(attname(:4), 'type') .EQ. 0 ) THEN
                memory( 1, mr_blk1(mr)) = vartype

             ELSE IF (STR_SAME(attname(:4), 'natt') .EQ. 0 ) THEN  ! natts
                memory( 1, mr_blk1(mr)) = nvatts
             ENDIF

           ELSE

* a regular attribute from the file or a definition

	      IF ( attype .EQ. NCCHAR ) THEN
* ... set up FORTRAN-independent pointer
	         CALL INIT_C_STRING_ARRAY( 1,
     .				     memory(1,mr_blk1(mr)),
     .				     mr_c_pointer(mr)    )

* ... string
	         CALL ATTRIB_STRING (cx, mr, dset, varid, varname, attname,
     .			          status)
	      ELSE
* ... array of numbers
	         CALL ATTRIB_VALS (memory( 1, mr_blk1(mr)), size, n, dset, 
     .                varid, varname, attname, status)
     
                 indx = 1
                 DO 800 i = i1, i2
                    memory(indx, mr_blk1(mr)) = memory(i,mr_blk1(mr))
                    indx = indx + 1
800              CONTINUE
                    

	      ENDIF
	      IF ( status .NE. ferr_ok ) GOTO 5000
           ENDIF

        ENDIF
* result is ready to return to the stack
	is_mr( isp ) = mr
        mr_bad_data(mr) = cx_bad_data(cx)

* got the requested component
	RETURN 2

! RETURN 1, which is to request a sub-component, is never used ...

* error exit
 5000	RETURN
	END	

