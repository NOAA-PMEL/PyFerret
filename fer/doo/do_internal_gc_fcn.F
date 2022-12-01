      SUBROUTINE DO_INTERNAL_GC_FCN( 
     .				fcn, num_com, com_cx,
     .				com1, mr1, pdim1, typ1,
     .				com2, mr2, pdim2, typ2,
     .				com3, mr3, pdim3, typ3,
     .				com4, mr4, pdim4, typ4,
     .				com5, mr5, pdim5, typ5,
     .				com6, mr6, pdim6, typ6,
     .				com7, mr7, pdim7, typ7,
     .				com8, mr8, pdim8, typ8,
     .				com9, mr9, pdim9, typ9,
     .				res, mres, pdimr, typr,
     .                          res_cx, status )

* This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
*
* execute a grid changing function

* the addition (or major modification) of an internal grid-changing function
* requires changes to these routines:
*	grid_chg_fcns.parm	- parameters to number the functions
*	XGRID_CHG_FCNS_DATA	- description of the function and args
*	DO_INTERNAL_GC_FCN	- code to execute the function
* (opt)	GCF_GET_IMPOSED_AXES	- return the "imposed" axes, if any
* (opt)	GCF_IMPOSE_RESULT_LIMS	- fill in limits of "imposed" axes, if any

* V450:  1/97 - new, based on operate.F, but for grid changing functions
*        7/97 - complete the ZAXREPLACE function
*       7/25/97 - increase number of args from 4 to 9
*               - add function PPLGRID
*       10/30/97 - fixed bug in UNRAVEL introduced in 7/25 changes
*                       "m5lox" --> "m10lox", etc.
*       11/12/97 - bug fix in ZAXREPLACE when destination field has missing
*                       values. New routine EXTRACT_Z_AXIS
* V491:  4/98 - fixed bugs in ZAXREPLACE: see err491_zaxreplace.jnl
*       need to guide calculation with  cx lims rather than mr lims of args
*       (see EXTRACT_Z_LINE) and
*       need to recognize the missing value flags of the first arg
* V500:  1/99 *sh* comandeered the TIMES2 function for string argument testing
*               renamed it to ECHO
*       2/99 changed UNRAVEL to XUNRAVEL. Added YUNRAVEL, ZUNRAVEL, TUNRAVEL
*       3/99 - made ECHO into a 2 argument function
* V510: 8/99 *sh* - added SAMPLE* functions (just copy arg 2 to result here --
*               the work is done in IS_ALGEBRA and IS_SAMPLE
* V533: *sh* 6/01 - arguments and result may be string or float
*                 - added SPAWN function
* V540: *sh* 12/01 - added STRCMP function
* V541: *kob* 6/02 - Add STRLEN, UPCASE,DNCASE,STRINDEX, STRRINDEX
*                        SUBSTRING, STRCAT, STRFLOAT 
*                    internal string handling functions   
* V552: *acm* 4/02 - Allow for zero-length strings in STRRINDEX fcn:
*                    STRRINDEX ("my string", "") had caused Ferret to hang.
* V580:10/04 *acm* Fix for bug 1037. In zaxreplace, before the loop calling 
*                  hunt_r4, set the initial guess for ndx_lo to 1 not 0. 
* V581  3/05 *acm* increase buffer length max_cmnd_len to 2048. 
*       3/05 *acm& Change call to strfloat, to return bad-flag if input 
*                  string cannot be converted to float.
* V600  5/06 *acm* In the fix to STRRINDEX, wrong indices in com2 
*                  CALL C_STRLEN(com2(1,i2,j2,k2,l2),ii) led to a crash
*                  when applying STRRINDEX to a list of strings.
* V68   1/12 *acm* call C_SUBSTR with integers for offset and length.
* V68  *acm* 1/12  ifdef double_p for double-precision ferret.
*       2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* V68   3/12 *KMS/ACM* see 6D branch: working on string bugs. Remove overloading
*                   of argument/return-type variable and predimensional (data size)
*                   array parameter (pointer size to one in double-precision Ferret).
* V68   5/12 *acm* Add ESEQUENCE, FSEQUENCE, SAMPLEM, SAMPLEN
* V693 10/14 *acm* ticket 2210: check for negative arguments to SUBSTRING
* keep max_cmnd_len for SPAWN at 2048 while Ferret command buffer
*      increases to 20480
* V702 1/17 *sh* for trac enhancement #2369 -- dynamic memory management 
* V74  3/18 *acm For issue #1199 disallow SPAWN in secure mode
* V75+ 4/19 *acm For DSG grids, call a routine to compute the *SEQUENCE fcns
* V75+ 4/19 *acm For RESHAPE to DSG grid, set the result context correctly
* v767 6/22 *acm* See issue 112: Detect whether 1-D grid that's the result of
*		  XSEQUENCE etc is not too large for 4-byte integer indexing
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
        include 'grid_chg_fcns.parm'
	include 'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xcontext.cmn'

* calling argument declarations:
	INTEGER	fcn, num_com, com_cx(9), status,
     .		mr1, mr2, mr3, mr4, mr5, mr6, mr7, mr8, mr9, mres,
     .          pdim1, pdim2, pdim3, pdim4, pdim5, pdim6, pdim7,
     .          pdim8, pdim9, pdimr,
     .          typ1, typ2, typ3, typ4, typ5, typ6, typ7, typ8, typ9,
     .          typr, res_cx
	REAL com1( pdim1,m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .             m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .       com2( pdim2,m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .             m2lot:m2hit,m2loe:m2hie,m2lof:m2hif ),
     .       com3( pdim3,m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,
     .             m3lot:m3hit,m3loe:m3hie,m3lof:m3hif ),
     .       com4( pdim4,m4lox:m4hix,m4loy:m4hiy,m4loz:m4hiz,
     .             m4lot:m4hit,m4loe:m4hie,m4lof:m4hif ),
     .       com5( pdim5,m5lox:m4hix,m5loy:m5hiy,m5loz:m5hiz,
     .             m5lot:m5hit,m5loe:m5hie,m5lof:m5hif ),
     .       com6( pdim6,m6lox:m4hix,m6loy:m6hiy,m6loz:m6hiz,
     .             m6lot:m6hit,m6loe:m6hie,m6lof:m6hif ),
     .       com7( pdim7,m7lox:m4hix,m7loy:m7hiy,m7loz:m7hiz,
     .             m7lot:m7hit,m7loe:m7hie,m7lof:m7hif ),
     .       com8( pdim8,m8lox:m4hix,m8loy:m8hiy,m8loz:m8hiz,
     .             m8lot:m8hit,m8loe:m8hie,m8lof:m8hif ),
     .       com9( pdim9,m9lox:m4hix,m9loy:m9hiy,m9loz:m9hiz,
     .             m9lot:m9hit,m9loe:m9hie,m9lof:m9hif ),
     .       res( pdimr,m10lox:m10hix,m10loy:m10hiy,m10loz:m10hiz,
     .            m10lot:m10hit,m10loe:m10hie,m10lof:m10hif )

* internal variable declarations:

	LOGICAL IS_SECURE, TM_ITSA_DSG
	INTEGER*8 CGRID_SIZE8, count, size
	INTEGER CX_DIM_LEN, TM_DSG_NFEATURES, TM_LENSTR1,
     .		icom, idim, ii,
     .		cx1, cx2, cx3,
     .           i,   j,   k,   l,   m,   n,
     .           i1,  j1,  k1,  l1,  m1,  n1,
     .           i2,  j2,  k2,  l2,  m2,  n2,
     .           i3,  j3,  k3,  l3,  m3,  n3,
     .          di1, dj1, dk1, dl1, dm1, dn1,
     .          di2, dj2, dk2, dl2, dm2, dn2,
     .          di3, dj3, dk3, dl3, dm3, dn3,
     .          si1, sj1, sk1, sl1, sm1, sn1,
     .          si2, sj2, sk2, sl2, sm2, sn2,
     .          si3, sj3, sk3, sl3, sm3, sn3,
     .          ind(nferdims),  grid, nfeatures, nres
        INTEGER klo, khi, ndx_lo, ioff, ilen
        INTEGER coded_pointer, uvar, item, start, end   ! for TIMES2
	REAL    bad_res, bad1, bad2, bad3,
     .          zval, frac
        REAL*8  TM_WORLD
        INTEGER max_cmnd_len
        PARAMETER ( max_cmnd_len = 2048 )
        CHARACTER GET_STRING_ELEMENT*(max_cmnd_len), buff*(max_cmnd_len)
	CHARACTER*48 TM_FMT, argval, fcn_name
        INTEGER nnew, slen, narg

* equivalence conveniences
* "inc" is the increments for each axis of each component
* "lo"  is the starting subscript (minus 1) for each axis of each component
	INTEGER	inc(nferdims,3), lo(nferdims,3)
	EQUIVALENCE
     .      (lo(1,1), si1), (lo(2,1), sj1), (lo(3,1), sk1),
     .      (lo(4,1), sl1), (lo(5,1), sm1), (lo(6,1), sn1),
     .      (lo(1,2), si2), (lo(2,2), sj2), (lo(3,2), sk2),
     .      (lo(4,2), sl2), (lo(5,2), sm2), (lo(6,2), sn2),
     .      (lo(1,3), si3), (lo(2,3), sj3), (lo(3,3), sk3),
     .      (lo(4,3), sl3), (lo(5,3), sm3), (lo(6,3), sn3),
     .      (inc(1,1),di1), (inc(2,1),dj1), (inc(3,1),dk1),
     .      (inc(4,1),dl1), (inc(5,1),dm1), (inc(6,1),dn1),
     .      (inc(1,2),di2), (inc(2,2),dj2), (inc(3,2),dk2),
     .      (inc(4,2),dl2), (inc(5,2),dm2), (inc(6,2),dn2),
     .      (inc(1,3),di3), (inc(2,3),dj3), (inc(3,3),dk3),
     .      (inc(4,3),dl3), (inc(5,3),dm3), (inc(6,3),dn3)

* initialization supported for up to three arguments; need to
* extend if a function is ever created with more than three
        IF ( num_com .GT. 3 ) THEN
           STOP '**ERROR: DO_INTERNAL_GC_FCN num_com > 3'
        ENDIF

* initialize commonly used variables
        status = ferr_ok
	cx1 = com_cx(1)
	cx2 = com_cx(2)
	cx3 = com_cx(3)

* flag(s) for bad or missing values
	bad1 = mr_bad_data ( mr1 )
	bad2 = mr_bad_data ( mr2 )
	bad3 = mr_bad_data ( mr3 )
	bad_res = mr_bad_data( mres )

* compute the delta increment for each axis of each component
	DO icom = 1, num_com
	DO idim = 1, nferdims
	   IF ( CX_DIM_LEN(idim,com_cx(icom)) .EQ. 1 ) THEN
	      inc(idim,icom) = 0
	   ELSE
	      inc(idim,icom) = 1
	   ENDIF
	ENDDO
	ENDDO

* Is the first argument a variable in a DSG grid? If so the *SEQUENCE 
* functions will apply any masking needed.

	grid = cx_grid(cx1)
	IF ( TM_ITSA_DSG(grid) ) nfeatures = TM_DSG_NFEATURES(grid)

* compute the starting subscript for each axis of each component
* (pre-decrement by 1 delta for looping ahead)
* NOTE: 
	DO icom = 1, num_com
	DO idim = 1, nferdims
 	   lo(idim,icom) = cx_lo_ss(com_cx(icom),idim) - inc(idim,icom)
	ENDDO
	ENDDO

* select the function
	GOTO ( 1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,
     .	       2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,
     .	       3100,3200,3300,3400,3500,3600 ) fcn


* "XSEQUENCE" - reshape/flatten the input into a line in X
 1100	IF (typr .EQ. pfcn_rtn_float) THEN

	  IF (TM_ITSA_DSG(grid)) THEN
	     nres = m10hix - m10lox + 1
	     CALL SEQUENCE_DSG (grid, com1, res, cx1, nfeatures, 
     .                          bad1, bad_res, nres)
	     RETURN
	  ENDIF

* Check that the result 1-D grid will not be too large to index
	  size = CGRID_SIZE8 (cx1)
	  IF (size .GT. arbitrary_large_int4) GOTO 5400

          i2 = m10lox
	  DO n = cx_lo_s6(cx1), cx_hi_s6(cx1)
	  DO m = cx_lo_s5(cx1), cx_hi_s5(cx1)
	  DO l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	  DO k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	  DO j = cx_lo_s2(cx1), cx_hi_s2(cx1)
	  DO i = cx_lo_s1(cx1), cx_hi_s1(cx1)
	     IF ( com1(1,i,j,k,l,m,n) .EQ. bad1 ) THEN
	        res(1,i2,m10loy,m10loz,m10lot,m10loe,m10lof)
     .             = bad_res
	     ELSE
	        res(1,i2,m10loy,m10loz,m10lot,m10loe,m10lof)
     .             = com1(1,i,j,k,l,m,n)
	     ENDIF
	     i2 = i2 + 1
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
        ELSE
          i2 = m10lox
	  DO n = cx_lo_s6(cx1), cx_hi_s6(cx1)
	  DO m = cx_lo_s5(cx1), cx_hi_s5(cx1)
	  DO l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	  DO k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	  DO j = cx_lo_s2(cx1), cx_hi_s2(cx1)
	  DO i = cx_lo_s1(cx1), cx_hi_s1(cx1)
	     CALL COPY_C_STRING(com1(1,i,j,k,l,m,n),
     .                 res(1,i2,m10loy,m10loz,m10lot,m10loe,m10lof))
	     i2 = i2 + 1
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
        ENDIF
	RETURN


* "TIMES2" (a.k.a. "ECHO") - debugging function
* ECHO the string that was passed in
* here is how you check the variable type
 1200   IF (mr_category(mr1) .NE. cat_string)
     .     WRITE (6,*) "ERROR: argument is not a string after all!"
* here is how you decode the string
        coded_pointer = mr_variable(mr1)
* ... note that the uvar number and item number of the string are
*     packed together in the coded_pointer value
        IF ( coded_pointer .GT. 0 ) THEN
* ... as of V5.0 should always come here
           uvar = coded_pointer/1000
           item = coded_pointer - 1000*uvar
           start = uvar_item_start( item, uvar )
           end   = uvar_item_end  ( item, uvar )
* ... echo string
           WRITE (6,'(1X,A)') 
     .          'You said: '//uvar_text(uvar)(start:end)
* ... compute result from arg 2
**** WARNING: This assumes com2 and res dimensions start and end at the same values ****
	   DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	   DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	   DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	   DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	   DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	      IF ( com2(1,i,j,k,l,m,n) .EQ. bad2 ) THEN
	         res(1,i,j,k,l,m,n) = bad_res
	      ELSE
	         res(1,i,j,k,l,m,n) = com2(1,i,j,k,l,m,n) * 2.0
	      ENDIF
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
        ELSE
* ... for future grids of strings
           STOP 'illegal string argument encoding'
!          pointer to array of strings encoded in memory(k,blk)
!          string = EXTRACT_FROM_STRING_GRID(memory(k,blk),i,j,k,l)
        ENDIF
        RETURN


* "RESHAPE" - reshape argument 1 into the grid of arg 2. Start output
* at the i,j,k,l,m,n point given in arg 2. It is commonly useful for taking
* monthly data and recasting it as a 12 month Z axis by a yearly axis
 1300   DO idim = 1, nferdims
 	   ind(idim) = cx_lo_ss(cx1,idim)
	ENDDO
	size = CGRID_SIZE8(cx1)
	count = 0	! total points xferred - used to prevent grid overflow

	IF (typr .EQ. pfcn_rtn_float) THEN

	  DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	  DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	  DO i = mr_lo_s1(mres), mr_hi_s1(mres)

	     count = count + 1
	     IF ( count .GT. size ) THEN
	        res(1,i,j,k,l,m,n) = bad_res
             ELSEIF ( com1(1,ind(1),ind(2),ind(3),ind(4),ind(5),ind(6))
     .                .EQ. bad1 ) THEN
	        res(1,i,j,k,l,m,n) = bad_res
	     ELSE
	        res(1,i,j,k,l,m,n) = 
     .                com1(1,ind(1),ind(2),ind(3),ind(4),ind(5),ind(6))
	     ENDIF

* ... increment the subscripts of the input argument
	     DO idim = 1, nferdims
	        IF (ind(idim) .LT. cx_hi_ss(cx1,idim) ) THEN
	           ind(idim) = ind(idim) + 1
	           EXIT
	        ELSE
	           ind(idim) =  cx_lo_ss(cx1,idim)
	        ENDIF
	     ENDDO

	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO

        ELSE

	  DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	  DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	  DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	  DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	  DO i = mr_lo_s1(mres), mr_hi_s1(mres)

	     count = count + 1
	     CALL COPY_C_STRING(
     .              com1(1,ind(1),ind(2),ind(3),ind(4),ind(5),ind(6)),
     .              res(1,i,j,k,l,m,n) )

* ... increment the subscripts of the input argument
	     DO idim = 1, nferdims
	        IF (ind(idim) .LT. cx_hi_ss(cx1,idim) ) THEN
	           ind(idim) = ind(idim) + 1
	           EXIT
	        ELSE
	           ind(idim) =  cx_lo_ss(cx1,idim)
	        ENDIF
	     ENDDO

	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO

        ENDIF

* Is this a reshape-to-DSGvar? Set the context to agree with arg2.

        grid = cx_grid(cx2)
        IF (TM_ITSA_DSG(grid)) CALL TRANSFER_CONTEXT (cx2, res_cx)

	RETURN


* "ZAXREPLACE(V,ZVALS,ZAX)" - regrid a sigma-coordinate-style (layered)
* variable, V, onto a depth axis, ZAX, guided by the depths in ZVALS
* ... sanity check -- V and ZVALS must match in vertical structure
 1400   IF ( (cx_lo_s3(cx1) .NE. cx_lo_s3(cx2))
     . .OR.  (cx_hi_s3(cx1) .NE. cx_hi_s3(cx2)) ) CALL ERRMSG
     .		(ferr_limits, status,
     .		 'Z limits of data and depth fields must match', *5000)

	IF (TM_ITSA_DSG( cx_grid(cx1) )) THEN
	   fcn_name = 'ZAXREPLACE'
	   buff = ' does not apply to Data from  '//
     .   'a DiscreteSampling Geometries (DSG) dataset.'
	   GOTO 5300
	ENDIF
 
* ... allocate a work array for a single profile of Z depths
        size = m2hiz-m2loz+1
	CALL GET_DYN_WORK_SPACE( size, status )
        IF ( status .NE. ferr_ok ) GOTO 5000


* ... loop over the X,Y,T,E,F range of the input fields
	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   l1 = l1 + dl1
	   l2 = l2 + dl2
	   j1 = sj1
	   j2 = sj2
	   DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	    j1 = j1 + dj1
	    j2 = j2 + dj2
	    i1 = si1
	    i2 = si2
	    DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	     i1 = i1 + di1
	     i2 = i2 + di2

* ... extract the vertical profile at this I,J,L location
             CALL EXTRACT_Z_LINE ( com2, bad2, i2, j2, l2, m2, n2,
     .               cx_lo_s3(cx2), cx_hi_s3(cx2), 
     .               m2lox, m2hix, m2loy, m2hiy, m2loz, m2hiz,
     .               m2lot, m2hit, m2loe, m2hie, m2lof, m2hif,
     .               workmem(plegacy_work_buffer)%ptr, klo, khi )

             IF ( klo .LT. 1 ) THEN

* ... no valid data in component 2 at this location
	        DO k = mr_lo_s3(mres), mr_hi_s3(mres)
		   res(1,i,j,k,l,m,n) = bad_res
	        ENDDO
             ELSE

* loop over the desired output Z-axis points
!               ndx_lo = 0    ! no guess for location of first depth
                ndx_lo = 1    ! fixes bug where if zval equals 1st point in axis
                              ! we got no match.
		DO k = mr_lo_s3(mres), mr_hi_s3(mres)

* ... next Z axis value (com3) to locate
                   zval = TM_WORLD(k,cx_grid(cx3),z_dim,box_middle)
* ... search the Z field (com2) for the index just below this value
* ... note that ndx_lo is always referenced to a starting index of 1
                   CALL HUNT_R4(workmem(plegacy_work_buffer)%ptr,
     .                          klo, khi, zval, ndx_lo, frac)
                   IF ( frac .LT. 0.0 ) THEN
* ... no luck -- the Z axis value we seek is outside the range in com2
                      res(1,i,j,k,l,m,n) = bad_res
                   ELSEIF ( com1(1,i1,j1,sk1+ndx_lo,l1,m1,n1)
     .                      .EQ. bad1 ) THEN
                      res(1,i,j,k,l,m,n) = bad_res
                   ELSEIF ( frac .EQ. 1.0 ) THEN      ! exactly on point
                      res(1,i,j,k,l,m,n) =
     .                   com1(1,i1,j1,sk1+ndx_lo,l1,m1,n1)
                   ELSEIF ( com1(1,i1,j1,sk1+ndx_lo+1,l1,m1,n1)
     .                      .EQ. bad1 ) THEN
                      res(1,i,j,k,l,m,n) = bad_res
                   ELSE
* ... interpolate data com1 field to get result
                      res(1,i,j,k,l,m,n) = 
     .                   frac  * com1(1,i1,j1,sk1+ndx_lo,  l1,m1,n1)
     .              + (1-frac) * com1(1,i1,j1,sk1+ndx_lo+1,l1,m1,n1)
                   ENDIF
		ENDDO

             ENDIF

	    ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO

        CALL RELEASE_DYN_WORK_SPACE
        RETURN


* "YSEQUENCE" - reshape/flatten the input into a line in Y
 1500	IF (typr .EQ. pfcn_rtn_float) THEN

	  IF (TM_ITSA_DSG(grid)) THEN
	  
	     nres = m10hiy - m10loy + 1
	     CALL SEQUENCE_DSG (grid, com1, res, cx1, nfeatures, 
     .                          bad1, bad_res, nres)
	     RETURN
	  ENDIF

* Check that the result 1-D grid will not be too large to index
	  size = CGRID_SIZE8 (cx1)
	  IF (size .GT. arbitrary_large_int4) GOTO 5400

	  j2 = m10loy

	  DO n = cx_lo_s6(cx1), cx_hi_s6(cx1)
	  DO m = cx_lo_s5(cx1), cx_hi_s5(cx1)
	  DO l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	  DO k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	  DO j = cx_lo_s2(cx1), cx_hi_s2(cx1)
	  DO i = cx_lo_s1(cx1), cx_hi_s1(cx1)
	     IF ( com1(1,i,j,k,l,m,n) .EQ. bad1 ) THEN
	        res(1,m10lox,j2,m10loz,m10lot,m10loe,m10lof)
     .             = bad_res
	     ELSE
	        res(1,m10lox,j2,m10loz,m10lot,m10loe,m10lof)
     .             = com1(1,i,j,k,l,m,n)
	     ENDIF
	     j2 = j2 + 1
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
        ELSE
	  j2 = m10loy
	  DO n = cx_lo_s6(cx1), cx_hi_s6(cx1)
	  DO m = cx_lo_s5(cx1), cx_hi_s5(cx1)
	  DO l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	  DO k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	  DO j = cx_lo_s2(cx1), cx_hi_s2(cx1)
	  DO i = cx_lo_s1(cx1), cx_hi_s1(cx1)
	     CALL COPY_C_STRING(com1(1,i,j,k,l,m,n),
     .                 res(1,m10lox,j2,m10loz,m10lot,m10loe,m10lof))
	     j2 = j2 + 1
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
        ENDIF
	RETURN


* "ZSEQUENCE" - reshape/flatten the input into a line in Z
 1600	IF (typr .EQ. pfcn_rtn_float) THEN

	  IF (TM_ITSA_DSG(grid)) THEN
	     nres = m10hiz - m10loz + 1
	     CALL SEQUENCE_DSG (grid, com1, res, cx1, nfeatures, 
     .                          bad1, bad_res, nres)
	     RETURN
	  ENDIF

* Check that the result 1-D grid will not be too large to index
	  size = CGRID_SIZE8 (cx1)
	  IF (size .GT. arbitrary_large_int4) GOTO 5400

	  k2 = m10loz
	  DO n = cx_lo_s6(cx1), cx_hi_s6(cx1)
	  DO m = cx_lo_s5(cx1), cx_hi_s5(cx1)
	  DO l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	  DO k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	  DO j = cx_lo_s2(cx1), cx_hi_s2(cx1)
	  DO i = cx_lo_s1(cx1), cx_hi_s1(cx1)
	     IF ( com1(1,i,j,k,l,m,n) .EQ. bad1 ) THEN
	        res(1,m10lox,m10loy,k2,m10lot,m10loe,m10lof)
     .             = bad_res
	     ELSE
	        res(1,m10lox,m10loy,k2,m10lot,m10loe,m10lof)
     .             = com1(1,i,j,k,l,m,n)
	     ENDIF
	     k2 = k2 + 1
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
        ELSE
	  k2 = m10loz
	  DO n = cx_lo_s6(cx1), cx_hi_s6(cx1)
	  DO m = cx_lo_s5(cx1), cx_hi_s5(cx1)
	  DO l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	  DO k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	  DO j = cx_lo_s2(cx1), cx_hi_s2(cx1)
	  DO i = cx_lo_s1(cx1), cx_hi_s1(cx1)
	     CALL COPY_C_STRING(com1(1,i,j,k,l,m,n),
     .                 res(1,m10lox,m10loy,k2,m10lot,m10loe,m10lof))
	     k2 = k2 + 1
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
        ENDIF
	RETURN


* "TSEQUENCE" - reshape/flatten the input into a line in T
 1700	IF (typr .EQ. pfcn_rtn_float) THEN

	  IF (TM_ITSA_DSG(grid)) THEN
	     nres = m10hit - m10lot + 1
	     CALL SEQUENCE_DSG (grid, com1, res, cx1, nfeatures, 
     .                          bad1, bad_res, nres)
	     RETURN
	  ENDIF

* Check that the result 1-D grid will not be too large to index
	  size = CGRID_SIZE8 (cx1)
	  IF (size .GT. arbitrary_large_int4) GOTO 5400

	  l2 = m10lot
	  DO n = cx_lo_s6(cx1), cx_hi_s6(cx1)
	  DO m = cx_lo_s5(cx1), cx_hi_s5(cx1)
	  DO l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	  DO k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	  DO j = cx_lo_s2(cx1), cx_hi_s2(cx1)
	  DO i = cx_lo_s1(cx1), cx_hi_s1(cx1)
	     IF ( com1(1,i,j,k,l,m,n) .EQ. bad1 ) THEN
	        res(1,m10lox,m10loy,m10loz,l2,m10loe,m10lof)
     .             = bad_res
	     ELSE
	        res(1,m10lox,m10loy,m10loz,l2,m10loe,m10lof)
     .             = com1(1,i,j,k,l,m,n)
	     ENDIF
	     l2 = l2 + 1
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
        ELSE
	  l2 = m10lot
	  DO n = cx_lo_s6(cx1), cx_hi_s6(cx1)
	  DO m = cx_lo_s5(cx1), cx_hi_s5(cx1)
	  DO l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	  DO k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	  DO j = cx_lo_s2(cx1), cx_hi_s2(cx1)
	  DO i = cx_lo_s1(cx1), cx_hi_s1(cx1)
	     CALL COPY_C_STRING(com1(1,i,j,k,l,m,n),
     .                 res(1,m10lox,m10loy,m10loz,l2,m10loe,m10lof))
	     l2 = l2 + 1
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
        ENDIF
	RETURN

	
* "ESEQUENCE" - reshape/flatten the input into a line in E
 1800	IF (typr .EQ. pfcn_rtn_float) THEN

	  IF (TM_ITSA_DSG(grid)) THEN
	     nres = m10hie - m10loe + 1
	     CALL SEQUENCE_DSG (grid, com1, res, cx1, nfeatures, 
     .                          bad1, bad_res, nres)
	     RETURN
	  ENDIF

* Check that the result 1-D grid will not be too large to index
	  size = CGRID_SIZE8 (cx1)
	  IF (size .GT. arbitrary_large_int4) GOTO 5400

	  m2 = m10loe
	  DO n = cx_lo_s6(cx1), cx_hi_s6(cx1)
	  DO m = cx_lo_s5(cx1), cx_hi_s5(cx1)
	  DO l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	  DO k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	  DO j = cx_lo_s2(cx1), cx_hi_s2(cx1)
	  DO i = cx_lo_s1(cx1), cx_hi_s1(cx1)
	     IF ( com1(1,i,j,k,l,m,n) .EQ. bad1 ) THEN
	        res(1,m10lox,m10loy,m10loz,m10lot,m2,m10lof)
     .             = bad_res
	     ELSE
	        res(1,m10lox,m10loy,m10loz,m10lot,m2,m10lof)
     .             = com1(1,i,j,k,l,m,n)
	     ENDIF
	     m2 = m2 + 1
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
        ELSE
	  m2 = m10loe
	  DO n = cx_lo_s6(cx1), cx_hi_s6(cx1)
	  DO m = cx_lo_s5(cx1), cx_hi_s5(cx1)
	  DO l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	  DO k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	  DO j = cx_lo_s2(cx1), cx_hi_s2(cx1)
	  DO i = cx_lo_s1(cx1), cx_hi_s1(cx1)
	     CALL COPY_C_STRING(com1(1,i,j,k,l,m,n),
     .                 res(1,m10lox,m10loy,m10loz,m10lot,m2,m10lof))
	     m2 = m2 + 1
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
        ENDIF
	RETURN

	
* "FSEQUENCE" - reshape/flatten the input into a line in F
 1900	IF (typr .EQ. pfcn_rtn_float) THEN

	  IF (TM_ITSA_DSG(grid)) THEN
	     nres = m10hif - m10lof + 1
	     CALL SEQUENCE_DSG (grid, com1, res, cx1, nfeatures, 
     .                          bad1, bad_res, nres)
	     RETURN
	  ENDIF

* Check that the result 1-D grid will not be too large to index
	  size = CGRID_SIZE8 (cx1)
	  IF (size .GT. arbitrary_large_int4) GOTO 5400

	  n2 = m10lof
	  DO n = cx_lo_s6(cx1), cx_hi_s6(cx1)
	  DO m = cx_lo_s5(cx1), cx_hi_s5(cx1)
	  DO l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	  DO k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	  DO j = cx_lo_s2(cx1), cx_hi_s2(cx1)
	  DO i = cx_lo_s1(cx1), cx_hi_s1(cx1)
	     IF ( com1(1,i,j,k,l,m,n) .EQ. bad1 ) THEN
	        res(1,m10lox,m10loy,m10loz,m10lot,m10loe,n2)
     .             = bad_res
	     ELSE
	        res(1,m10lox,m10loy,m10loz,m10lot,m10loe,n2)
     .             = com1(1,i,j,k,l,m,n)
	     ENDIF
	     n2 = n2 + 1
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
        ELSE
	  n2 = m10lof
	  DO n = cx_lo_s6(cx1), cx_hi_s6(cx1)
	  DO m = cx_lo_s5(cx1), cx_hi_s5(cx1)
	  DO l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	  DO k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	  DO j = cx_lo_s2(cx1), cx_hi_s2(cx1)
	  DO i = cx_lo_s1(cx1), cx_hi_s1(cx1)
	     CALL COPY_C_STRING(com1(1,i,j,k,l,m,n),
     .                 res(1,m10lox,m10loy,m10loz,m10lot,m10loe,n2))
	     n2 = n2 + 1
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
        ENDIF
	RETURN


* "SAMPLEI", "SAMPLEJ", "SAMPLEK", "SAMPLEL" , "SAMPLEM", "SAMPLEN"- 
* just copy arg 2 to result
* setup for sampling done elsewhere
 2000   CONTINUE        ! SAMPLEI
 2100   CONTINUE        ! SAMPLEJ
 2200   CONTINUE        ! SAMPLEK
 2300   CONTINUE        ! SAMPLEL
 2400   CONTINUE        ! SAMPLEN
 2500   CONTINUE        ! SAMPLEN
	
	IF (TM_ITSA_DSG( cx_grid(cx1) )) THEN
	   fcn_name = 'SAMPLEI'
	   IF (fcn .EQ. 2100) fcn_name = 'SAMPLEJ'
	   IF (fcn .EQ. 2100) fcn_name = 'SAMPLEJ'
	   IF (fcn .EQ. 2200) fcn_name = 'SAMPLEK'
	   IF (fcn .EQ. 2300) fcn_name = 'SAMPLEL'
	   IF (fcn .EQ. 2400) fcn_name = 'SAMPLEM'
	   IF (fcn .EQ. 2500) fcn_name = 'SAMPLEN'
	   
	   buff = ' does not apply to Data from a '//
     .   'Discrete Sampling Geometries (DSG) dataset. Use masking'
	   GOTO 5300

	ENDIF

	IF (typr .EQ. pfcn_rtn_float) THEN
**** WARNING: This assumes com2 and res dimensions start and end at the same values ****
	  DO n = cx_lo_s6(cx2), cx_hi_s6(cx2)
	  DO m = cx_lo_s5(cx2), cx_hi_s5(cx2)
	  DO l = cx_lo_s4(cx2), cx_hi_s4(cx2)
	  DO k = cx_lo_s3(cx2), cx_hi_s3(cx2)
	  DO j = cx_lo_s2(cx2), cx_hi_s2(cx2)
	  DO i = cx_lo_s1(cx2), cx_hi_s1(cx2)
	     IF ( com2(1,i,j,k,l,m,n) .EQ. bad2 ) THEN
	        res(1,i,j,k,l,m,n) = bad_res
	     ELSE
	        res(1,i,j,k,l,m,n) = com2(1,i,j,k,l,m,n)
	     ENDIF
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
        ELSE
**** WARNING: This assumes com2 and res dimensions start and end at the same values ****
	  DO n = cx_lo_s6(cx2), cx_hi_s6(cx2)
	  DO m = cx_lo_s5(cx2), cx_hi_s5(cx2)
	  DO l = cx_lo_s4(cx2), cx_hi_s4(cx2)
	  DO k = cx_lo_s3(cx2), cx_hi_s3(cx2)
	  DO j = cx_lo_s2(cx2), cx_hi_s2(cx2)
	  DO i = cx_lo_s1(cx2), cx_hi_s1(cx2)
	     CALL COPY_C_STRING(com2(1,i,j,k,l,m,n),
     .                           res(1,i,j,k,l,m,n))
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
	  ENDDO
        ENDIF
	RETURN


* SPAWN function - perform a Unix system command
 2600	CONTINUE

*spawn is disallowed in secure mode.
        IF (IS_SECURE()) GOTO 5200

* ... get the command string to be executed

	IF (CGRID_SIZE8(cx1) .NE. 1) CALL ERRMSG(ferr_invalid_command,
     .      status, 'SPAWN can only execute a single command', *5000)
        buff = GET_STRING_ELEMENT(m1lox, m1hix, m1loy, m1hiy,
     .              m1loz, m1hiz, m1lot, m1hit, m1loe, m1hie,
     .              m1lof, m1hif,  com1, m1lox, m1loy, m1loz,
     .              m1lot, m1loe, m1lof, max_cmnd_len, slen)
        IF (slen .EQ. max_cmnd_len) CALL ERRMSG(ferr_invalid_command,
     .       status, 'SPAWN command too long -- 2048 char max', *5000)
* ... execute the command and store the results
        CALL STORE_SYS_STRINGS(buff, res_cx, mres, 1, nnew, status)
        RETURN


* STRCMP function
 2700	CONTINUE
	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   l1 = l1 + dl1
	   l2 = l2 + dl2
	   k1 = sk1
	   k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    k1 = k1 + dk1
	    k2 = k2 + dk2
	    j1 = sj1
	    j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     j1 = j1 + dj1
	     j2 = j2 + dj2
	     i1 = si1
	     i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	      i1 = i1 + di1
	      i2 = i2 + di2
              CALL C_STRCMP(com1(1,i1,j1,k1,l1,m1,n1),
     .                      com2(1,i2,j2,k2,l2,m2,n2), ii)
              res(1,i,j,k,l,m,n) = ii
	     ENDDO
	    ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO
        RETURN


* *kob* 3/02 STRLEN function
 2800	CONTINUE
	n1 = sn1
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 m1 = sm1
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  l1 = sl1
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   l1 = l1 + dl1
	   k1 = sk1
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    k1 = k1 + dk1
	    j1 = sj1
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     j1 = j1 + dj1
	     i1 = si1
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	      i1 = i1 + di1
              CALL C_STRLEN(com1(1,i1,j1,k1,l1,m1,n1), ii)
              res(1,i,j,k,l,m,n) = ii
	     ENDDO
	    ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO
        RETURN

* *kob* 3/02 UPCASE function
 2900	CONTINUE
	n1 = sn1
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 m1 = sm1
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  l1 = sl1
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   l1 = l1 + dl1
	   k1 = sk1
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    k1 = k1 + dk1
	    j1 = sj1
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     j1 = j1 + dj1
	     i1 = si1
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	      i1 = i1 + di1
              CALL C_UPCASE(com1(1,i1,j1,k1,l1,m1,n1),
     .                      res(1,i,j,k,l,m,n))
	     ENDDO
	    ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO
        RETURN


* *kob* 3/02 STRINDEX function
 3000	CONTINUE
	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   l1 = l1 + dl1
	   l2 = l2 + dl2
	   k1 = sk1
	   k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    k1 = k1 + dk1
	    k2 = k2 + dk2
	    j1 = sj1
	    j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     j1 = j1 + dj1
	     j2 = j2 + dj2
	     i1 = si1
	     i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	      i1 = i1 + di1
	      i2 = i2 + di2
              CALL C_STRINDEX(com1(1,i1,j1,k1,l1,m1,n1),
     .                        com2(1,i2,j2,k2,l2,m2,n2), ii)
              res(1,i,j,k,l,m,n) = ii
	     ENDDO
	    ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO
        RETURN

* *kob* 3/02 STRRINDEX function
 3100	CONTINUE
	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   l1 = l1 + dl1
	   l2 = l2 + dl2
	   k1 = sk1
	   k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    k1 = k1 + dk1
	    k2 = k2 + dk2
	    j1 = sj1
	    j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     j1 = j1 + dj1
	     j2 = j2 + dj2
	     i1 = si1
	     i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	      i1 = i1 + di1
	      i2 = i2 + di2
              CALL C_STRLEN(com2(1,i2,j2,k2,l2,m2,n2), ii)
              IF (ii .LE. 0) THEN
                 CALL C_STRLEN(com1(1,i1,j1,k1,l1,m1,n1), ii)
                 res(1,i,j,k,l,m,n) = ii
              ELSE
                 CALL C_STRRINDEX(com1(1,i1,j1,k1,l1,m1,n1),
     .                            com2(1,i2,j2,k2,l2,m2,n2), ii)
                 res(1,i,j,k,l,m,n) = ii
              ENDIF
	     ENDDO
	    ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO
        RETURN

* *kob* 3/02 DNCASE function
 3200	CONTINUE
	n1 = sn1
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 m1 = sm1
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  l1 = sl1
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   l1 = l1 + dl1
	   k1 = sk1
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    k1 = k1 + dk1
	    j1 = sj1
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     j1 = j1 + dj1
	     i1 = si1
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	      i1 = i1 + di1
              CALL C_DNCASE(com1(1,i1,j1,k1,l1,m1,n1),
     .                      res(1,i,j,k,l,m,n))
	     ENDDO
	    ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO
        RETURN

 3300	CONTINUE
* *kob* STRCAT function 3/02
	n1 = sn1
	n2 = sn2
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 m1 = sm1
	 m2 = sm2
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  l1 = sl1
	  l2 = sl2
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   l1 = l1 + dl1
	   l2 = l2 + dl2
	   k1 = sk1
	   k2 = sk2
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    k1 = k1 + dk1
	    k2 = k2 + dk2
	    j1 = sj1
	    j2 = sj2
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     j1 = j1 + dj1
	     j2 = j2 + dj2
	     i1 = si1
	     i2 = si2
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	      i1 = i1 + di1
	      i2 = i2 + di2
              CALL C_STRCAT(com1(1,i1,j1,k1,l1,m1,n1),
     .                      com2(1,i2,j2,k2,l2,m2,n2),
     .                      res(1,i,j,k,l,m,n))
	     ENDDO
	    ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO
        RETURN

 3400	CONTINUE
* *kob* SUBSTRING function 3/02
	n1 = sn1
	n2 = sn2
	n3 = sn3
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 n2 = n2 + dn2
	 n3 = n3 + dn3
	 m1 = sm1
	 m2 = sm2
	 m3 = sm3
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  m2 = m2 + dm2
	  m3 = m3 + dm3
	  l1 = sl1
	  l2 = sl2
	  l3 = sl3
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   l1 = l1 + dl1
	   l2 = l2 + dl2
	   l3 = l3 + dl3
	   k1 = sk1
	   k2 = sk2
	   k3 = sk3
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    k1 = k1 + dk1
	    k2 = k2 + dk2
	    k3 = k3 + dk3
	    j1 = sj1
	    j2 = sj2
	    j3 = sj3
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     j1 = j1 + dj1
	     j2 = j2 + dj2
	     j3 = j3 + dj3
	     i1 = si1
	     i2 = si2
	     i3 = si3
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	      i1 = i1 + di1
	      i2 = i2 + di2
	      i3 = i3 + di3
* Convert the offset and length to integers
              ioff = com2(1,i2,j2,k2,l2,m2,n2)
              ilen = com3(1,i3,j3,k3,l3,m3,n3)
* argument checking:
	      IF (ioff.LT.0) THEN
	         narg = 2
		 GOTO 5100
	      ENDIF
	      IF (ilen.LT.0)  THEN
	         narg = 3
		 GOTO 5100
	      ENDIF
              CALL C_SUBSTR(com1(1,i1,j1,k1,l1,m1,n1),
     .                      ioff, ilen, res(1,i,j,k,l,m,n))
	     ENDDO
	    ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO
        RETURN

* *kob* 5/02 STRFLOAT function
 3500	CONTINUE
	n1 = sn1
	DO n = mr_lo_s6(mres), mr_hi_s6(mres)
	 n1 = n1 + dn1
	 m1 = sm1
	 DO m = mr_lo_s5(mres), mr_hi_s5(mres)
	  m1 = m1 + dm1
	  l1 = sl1
	  DO l = mr_lo_s4(mres), mr_hi_s4(mres)
	   l1 = l1 + dl1
	   k1 = sk1
	   DO k = mr_lo_s3(mres), mr_hi_s3(mres)
	    k1 = k1 + dk1
	    j1 = sj1
	    DO j = mr_lo_s2(mres), mr_hi_s2(mres)
	     j1 = j1 + dj1
	     i1 = si1
	     DO i = mr_lo_s1(mres), mr_hi_s1(mres)
	      i1 = i1 + di1
              CALL C_STRFLOAT(com1(1,i1,j1,k1,l1,m1,n1),
     .                        res(1,i,j,k,l,m,n), bad_res)
	     ENDDO
	    ENDDO
	   ENDDO
	  ENDDO
	 ENDDO
	ENDDO
        RETURN


* Undefined functions
 3600   CONTINUE
	STOP '**ERROR: GC FUNCTION NOT IMPLEMENTED'


* error exits
 5000	RETURN
 5100	argval = TM_FMT(DBLE(narg), 0, 8, slen)
	CALL ERRMSG(ferr_out_of_range, status,
     .   'Negative argument to function. Argument '//
     .    argval(:slen)//' : SUBSTRING', *5000 )

 5200	CALL ERRMSG(ferr_invalid_command, status,
     .   'SPAWN not allowed in secure mode', *5000 )

 5300	CALL ERRMSG(ferr_invalid_command, status,
     .   fcn_name(:TM_LENSTR1(fcn_name)) //' '//
     .   buff(:TM_LENSTR1(buff)), *5000 )


 5400   CONTINUE
	CALL ERRMSG(ferr_grid_definition, status,
     .     'Result 1-d grid too large for 4-byte integer indexing: '//
     .   fcn_name(:TM_LENSTR1(fcn_name)) //' '//
     .   buff(:TM_LENSTR1(buff)), *5000 )

	END
