      SUBROUTINE DO_INTERNAL_GC_FCN( memory,
     .				fcn, num_com, com_cx,
     .				com1, mr1, typ1,
     .				com2, mr2, typ2,
     .				com3, mr3, typ3,
     .				com4, mr4, typ4,
     .				com5, mr5, typ5,
     .				com6, mr6, typ6,
     .				com7, mr7, typ7,
     .				com8, mr8, typ8,
     .				com9, mr9, typ9,
     .				res, mres, typr,
     .                          res_cx, status )

* This software was developed by the Thermal Modeling and Analysis
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

	include	'ferret.parm'
	include	'errmsg.parm'
	include 'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xcontext.cmn'

* calling argument declarations:
	INTEGER	fcn, num_com, com_cx(9), status,
     .		mr1, mr2, mr3, mr4, mr5, mr6, mr7, mr8, mr9, mres,
     .          typ1, typ2,typ3,typ4,typ5,typ6,typ7,typ8,typ9,
     .          typr, res_cx
        REAL	memory( mem_blk_size, max_mem_blks )
	REAL    com1(typ1,m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit),
     .		com2(typ2,m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit),
     .		com3(typ3,m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit),
     .		com4(typ4,m4lox:m4hix,m4loy:m4hiy,m4loz:m4hiz,m4lot:m4hit),
     .		com5(typ5,m5lox:m4hix,m5loy:m5hiy,m5loz:m5hiz,m5lot:m5hit),
     .		com6(typ6,m6lox:m4hix,m6loy:m6hiy,m6loz:m6hiz,m6lot:m6hit),
     .		com7(typ7,m7lox:m4hix,m7loy:m7hiy,m7loz:m7hiz,m7lot:m7hit),
     .		com8(typ8,m8lox:m4hix,m8loy:m8hiy,m8loz:m8hiz,m8lot:m8hit),
     .		com9(typ9,m9lox:m4hix,m9loy:m9hiy,m9loz:m9hiz,m9lot:m9hit),
     .		 res(typr,m10lox:m10hix,m10loy:m10hiy,
     .					  m10loz:m10hiz,m10lot:m10hit)

* internal variable declarations:
	INTEGER CX_DIM_LEN,  CGRID_SIZE, count, size,
     .		i, j, k, l, icom, idim, ii,
     .		cx1, cx2, cx3, cx4, cx5, cx6, cx7, cx8, cx9,
     .		i1, j1, k1, l1, i2, j2, k2, l2, i3,j3,k3,l3,
     .		di1, dj1, dk1, dl1, di2, dj2, dk2, dl2,
     .		di3, dj3, dk3, dl3, di4, dj4, dk4, dl4,
     .		di5, dj5, dk5, dl5, di6, dj6, dk6, dl6,
     .		si1, sj1, sk1, sl1, si2, sj2, sk2, sl2,
     .		si3, sj3, sk3, sl3, si4, sj4, sk4, sl4,
     .		si5, sj5, sk5, sl5, si6, sj6, sk6, sl6
!     .		i3, j3, k3, l3, i4, j4, k4, l4,
!     .		i5, j5, k5, l5, i6, j6, k6, l6,
        INTEGER klo, khi, ndx_lo, wkblk
        INTEGER coded_pointer, uvar, item, start, end   ! for TIMES2

	REAL    bad_res,
     .          bad1, bad2, bad3, bad4, bad5, bad6, bad7, bad8, bad9
        REAL    zval, frac
        REAL*8  TM_WORLD

        INTEGER max_cmnd_len
        PARAMETER ( max_cmnd_len = 500)
        CHARACTER GET_STRING_ELEMENT*(max_cmnd_len), buff*(max_cmnd_len)
        INTEGER nnew, slen


* equivalence conveniences
* "inc" is the increments for each axis of each component
* "lo"  is the starting subscript (minus 1) for each axis of each component
	INTEGER	inc(4,6), lo(4,6), ind(4)
	EQUIVALENCE
     .	(lo(1,1),si1), (lo(2,1),sj1), (lo(3,1),sk1), (lo(4,1),sl1),
     .	(lo(1,2),si2), (lo(2,2),sj2), (lo(3,2),sk2), (lo(4,2),sl2),
     .	(lo(1,3),si3), (lo(2,3),sj3), (lo(3,3),sk3), (lo(4,3),sl3),
     .	(lo(1,4),si4), (lo(2,4),sj4), (lo(3,4),sk4), (lo(4,4),sl4),
     .	(lo(1,5),si5), (lo(2,5),sj5), (lo(3,5),sk5), (lo(4,5),sl5),
     .	(lo(1,6),si6), (lo(2,6),sj6), (lo(3,6),sk6), (lo(4,6),sl6),
     .	(inc(1,1),di1),(inc(2,1),dj1),(inc(3,1),dk1),(inc(4,1),dl1),
     .	(inc(1,2),di2),(inc(2,2),dj2),(inc(3,2),dk2),(inc(4,2),dl2),
     .	(inc(1,3),di3),(inc(2,3),dj3),(inc(3,3),dk3),(inc(4,3),dl3),
     .	(inc(1,4),di4),(inc(2,4),dj4),(inc(3,4),dk4),(inc(4,4),dl4),
     .	(inc(1,5),di5),(inc(2,5),dj5),(inc(3,5),dk5),(inc(4,5),dl5),
     .	(inc(1,6),di6),(inc(2,6),dj6),(inc(3,6),dk6),(inc(4,6),dl6),
     .	(ind(1),i2),   (ind(2),j2),   (ind(3),k2),   (ind(4),l2)

* internal parameter declarations:
	REAL*8          pdays_by_1900
	PARAMETER     ( pdays_by_1900 = 59958230400.0 / (60.*60.*24.) )
	INTEGER	    pfloat_words, pptr_words
	PARAMETER  (pfloat_words = 1,
     .		    pptr_words   = 2 )

* initialize commonly used variables
        status = ferr_ok
	cx1 = com_cx(1)
	cx2 = com_cx(2)
	cx3 = com_cx(3)
	cx4 = com_cx(4)
	cx5 = com_cx(5)
	cx6 = com_cx(6)
	cx7 = com_cx(7)
	cx8 = com_cx(8)
	cx9 = com_cx(9)

* flag(s) for bad or missing values
	bad1 = mr_bad_data ( mr1 )
	bad2 = mr_bad_data ( mr2 )
	bad3 = mr_bad_data ( mr3 )
	bad4 = mr_bad_data ( mr4 )
	bad5 = mr_bad_data ( mr5 )
	bad6 = mr_bad_data ( mr6 )
	bad7 = mr_bad_data ( mr7 )
	bad8 = mr_bad_data ( mr8 )
	bad9 = mr_bad_data ( mr9 )
	bad_res = mr_bad_data( mres )

* compute the delta increment for each axis of each component
	DO 10 icom = 1, num_com
	DO 10 idim = 1, 4
	   IF ( CX_DIM_LEN(idim,com_cx(icom)) .EQ. 1 ) THEN
	      inc(idim,icom) = 0
	   ELSE
	      inc(idim,icom) = 1
	   ENDIF
 10	CONTINUE

* compute the starting subscript for each axis of each component
* (pre-decrement by 1 delta for looping ahead)
* NOTE: 
	DO 20 icom = 1, num_com
	DO 20 idim = 1, 4
 20	lo(idim,icom) = cx_lo_ss(com_cx(icom),idim) - inc(idim,icom)

* select the function
	GOTO (  1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,
     .		2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,
     .		3100,3200,3300,3400,3500			  ) fcn

* "XSEQUENCE" - reshape/flatten the input into a line in X
 1100	IF (typr .EQ. pfloat_words) THEN
          i2 = m10lox
	  DO 1110 l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	   DO 1110 k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	    DO 1110 j = cx_lo_s2(cx1), cx_hi_s2(cx1)
	     DO 1110 i = cx_lo_s1(cx1), cx_hi_s1(cx1)

	     IF ( com1(1,i,j,k,l) .EQ. bad1 ) THEN
	        res(1,i2,m10loy,m10loz,m10lot) = bad_res
	     ELSE
	        res(1,i2,m10loy,m10loz,m10lot) = com1(1,i,j,k,l)
	     ENDIF
	     i2 = i2 + 1

 1110     CONTINUE
        ELSE
          i2 = m10lox
	  DO 1160 l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	   DO 1160 k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	    DO 1160 j = cx_lo_s2(cx1), cx_hi_s2(cx1)
	     DO 1160 i = cx_lo_s1(cx1), cx_hi_s1(cx1)

	        CALL COPY_C_STRING(com1(1,i,j,k,l),
     .                             res(1,i2,m10loy,m10loz,m10lot))

	     i2 = i2 + 1
 1160	  CONTINUE
        ENDIF
	RETURN

* "TIMES2" (a.k.a. "ECHO") - debugging function

* ECHO the string that was passed in

* here's how you check the variable type
 1200   IF (mr_category(mr1) .NE. cat_string)
     .     WRITE (6,*) "ERROR: argument is not a string after all!"

* here's how you decode the string
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
	   DO 1205 l = mr_lo_s4(mres), mr_hi_s4(mres)
	    DO 1205 k = mr_lo_s3(mres), mr_hi_s3(mres)
	      DO 1205 j = mr_lo_s2(mres), mr_hi_s2(mres)
	       DO 1205 i = mr_lo_s1(mres), mr_hi_s1(mres)
	         IF ( com2(1,i,j,k,l) .EQ. bad2 ) THEN
	             res(1,i,j,k,l) = bad_res
	         ELSE
	             res(1,i,j,k,l) = com2(1,i,j,k,l) * 2.0
	         ENDIF
 1205      CONTINUE

        ELSE
* ... for future grids of strings
           STOP 'illegal string argument encoding'
!          pointer to array of strings encoded in memory(k,blk)
!          string = EXTRACT_FROM_STRING_GRID(memory(k,blk),i,j,k,l)
        ENDIF
        RETURN

* "RESHAPE" - reshape argument 1 into the grid of arg 2. Start output
* at the i,j,k,l point given in arg 2. It is commonly useful for taking
* monthly data and recasting it as a 12 month Z axis by a yearly axis
 1300      DO 1310 idim = 1, 4
 1310	ind(idim) = cx_lo_ss(cx1,idim) !i2,j2,k2,l2 <-- comp 1 start indices
	size = CGRID_SIZE(cx1)
	count = 0	! total points xferred - used to prevent grid overflow

	IF (typr .EQ. pfloat_words) THEN

	  DO 1350 l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO 1350 k = mr_lo_s3(mres), mr_hi_s3(mres)
	    DO 1350 j = mr_lo_s2(mres), mr_hi_s2(mres)
	     DO 1350 i = mr_lo_s1(mres), mr_hi_s1(mres)

	     count = count + 1
	     IF ( count .GT. size ) THEN
	        res(1,i,j,k,l) = bad_res
	     ELSEIF ( com1(1,i2,j2,k2,l2) .EQ. bad1 ) THEN
	        res(1,i,j,k,l) = bad_res
	     ELSE
	        res(1,i,j,k,l) = com1(1,i2,j2,k2,l2)
	     ENDIF

* ... increment the subscripts of the input argument
	     DO 1320 idim = 1, 4
	        IF (ind(idim) .LT. cx_hi_ss(cx1,idim) ) THEN
	           ind(idim) = ind(idim) + 1
	           GOTO 1350
	        ELSE
	           ind(idim) =  cx_lo_ss(cx1,idim)
	        ENDIF
 1320	     CONTINUE

 1350	  CONTINUE

        ELSE

	  DO 1390 l = mr_lo_s4(mres), mr_hi_s4(mres)
	   DO 1390 k = mr_lo_s3(mres), mr_hi_s3(mres)
	    DO 1390 j = mr_lo_s2(mres), mr_hi_s2(mres)
	     DO 1390 i = mr_lo_s1(mres), mr_hi_s1(mres)

	     count = count + 1
	     CALL COPY_C_STRING(com1(1,i2,j2,k2,l2), res(1,i,j,k,l))

* ... increment the subscripts of the input argument
	     DO 1360 idim = 1, 4
	        IF (ind(idim) .LT. cx_hi_ss(cx1,idim) ) THEN
	           ind(idim) = ind(idim) + 1
	           GOTO 1390
	        ELSE
	           ind(idim) =  cx_lo_ss(cx1,idim)
	        ENDIF
 1360	     CONTINUE

 1390	  CONTINUE
        ENDIF
	RETURN

* "ZAXREPLACE(V,ZVALS,ZAX)" - regrid a sigma-coordinate-style (layered)
* variable, V, onto a depth axis, ZAX, guided by the depths in ZVALS
* ... sanity check -- V and ZVALS must match in vertical structure
 1400   IF ( (cx_lo_s3(cx1) .NE. cx_lo_s3(cx2))
     . .OR.  (cx_hi_s3(cx1) .NE. cx_hi_s3(cx2)) ) CALL ERRMSG
     .		(ferr_limits, status,
     .		 'Z limits of data and depth fields must match', *5000)

* ... allocate a work array for a single profile of Z depths
	CALL GET_WORK_SPC( m2hiz-m2loz+1, wkblk, status )
        IF ( status .NE. ferr_ok ) GOTO 5000

* ... loop over the X,Y,T range of the input fields
	l1 = sl1
	l2 = sl2
	DO 1450 l = mr_lo_s4(mres), mr_hi_s4(mres)
	l1 = l1 + dl1
	l2 = l2 + dl2
	j1 = sj1
	j2 = sj2
	 DO 1450 j = mr_lo_s2(mres), mr_hi_s2(mres)
	 j1 = j1 + dj1
	 j2 = j2 + dj2
	 i1 = si1
	 i2 = si2
	  DO 1450 i = mr_lo_s1(mres), mr_hi_s1(mres)
	  i1 = i1 + di1
	  i2 = i2 + di2

* ... extract the vertical profile at this I,J,L location
          CALL EXTRACT_Z_LINE ( com2, bad2, i2, j2, l2,
     .                  cx_lo_s3(cx2), cx_hi_s3(cx2), 
     .          	m2lox,m2hix,m2loy,m2hiy,m2loz,m2hiz,m2lot,m2hit,
     .                  memory(1,wkblk), klo, khi )

          IF ( klo .LT. 1 ) THEN
* ... no valid data in component 2 at this location
	    DO 1405 k = mr_lo_s3(mres), mr_hi_s3(mres)
 1405          res(1,i,j,k,l) = bad_res

          ELSE

* loop over the desired output Z-axis points
            ndx_lo = 0    ! no guess for location of first depth
            ndx_lo = 1    ! fixes bug where if zval equals 1st point in axis
                          ! we got no match.

	    DO 1410 k = mr_lo_s3(mres), mr_hi_s3(mres)

* ... next Z axis value (com3) to locate
             zval = SNGL(TM_WORLD(k,cx_grid(cx3),z_dim,box_middle))

* ... search the Z field (com2) for the index just below this value
* ... note that ndx_lo is always referenced to a starting index of 1
	     CALL HUNT_R4(memory(1,wkblk), klo, khi, zval, ndx_lo, frac)

             IF ( frac .LT. 0.0 ) THEN
* ... no luck -- the Z axis value we seek is outside the range in com2
                res(1,i,j,k,l) = bad_res
             ELSEIF ( com1(1,i1,j1,sk1+ndx_lo,l1) .EQ. bad1 ) THEN
                res(1,i,j,k,l) = bad_res
             ELSEIF ( frac .EQ. 1.0 ) THEN                  ! exactly on point
	        res(1,i,j,k,l) =  com1(1,i1,j1,sk1+ndx_lo,l1)
             ELSEIF ( com1(1,i1,j1,sk1+ndx_lo+1,l1) .EQ. bad1 ) THEN
                res(1,i,j,k,l) = bad_res
             ELSE
* ... interpolate data com1 field to get result
	        res(1,i,j,k,l) =  frac   * com1(1,i1,j1,sk1+ndx_lo,  l1)
     .			     + (1-frac)* com1(1,i1,j1,sk1+ndx_lo+1,l1)
             ENDIF
 1410	    CONTINUE

           ENDIF

 1450	CONTINUE
        CALL RELEASE_WORK_SPC
        RETURN

* "YSEQUENCE" - reshape/flatten the input into a line in Y
 1500	IF (typr .EQ. pfloat_words) THEN
	  i2 = m10loy
	  DO 1510 l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	   DO 1510 k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	    DO 1510 j = cx_lo_s2(cx1), cx_hi_s2(cx1)
	     DO 1510 i = cx_lo_s1(cx1), cx_hi_s1(cx1)
 
	     IF ( com1(1,i,j,k,l) .EQ. bad1 ) THEN
	        res(1,m10lox,i2,m10loz,m10lot) = bad_res
	     ELSE
	        res(1,m10lox,i2,m10loz,m10lot) = com1(1,i,j,k,l)
	     ENDIF
	     i2 = i2 + 1

 1510	  CONTINUE
        ELSE
	  i2 = m10loy
	  DO 1560 l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	   DO 1560 k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	    DO 1560 j = cx_lo_s2(cx1), cx_hi_s2(cx1)
	     DO 1560 i = cx_lo_s1(cx1), cx_hi_s1(cx1)
 
	        CALL COPY_C_STRING(com1(1,i,j,k,l),
     .                             res(1,m10lox,i2,m10loz,m10lot))

	     i2 = i2 + 1

 1560	  CONTINUE
        ENDIF

	RETURN


* "ZSEQUENCE" - reshape/flatten the input into a line in Z
 1600	IF (typr .EQ. pfloat_words) THEN
	  i2 = m10loz
	  DO 1610 l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	   DO 1610 k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	    DO 1610 j = cx_lo_s2(cx1), cx_hi_s2(cx1)
	     DO 1610 i = cx_lo_s1(cx1), cx_hi_s1(cx1)

	     IF ( com1(1,i,j,k,l) .EQ. bad1 ) THEN
	        res(1,m10lox,m10loy,i2,m10lot) = bad_res
	     ELSE
	        res(1,m10lox,m10loy,i2,m10lot) = com1(1,i,j,k,l)
	     ENDIF
	     i2 = i2 + 1

 1610	  CONTINUE
        ELSE
	  i2 = m10loz
	  DO 1660 l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	   DO 1660 k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	    DO 1660 j = cx_lo_s2(cx1), cx_hi_s2(cx1)
	     DO 1660 i = cx_lo_s1(cx1), cx_hi_s1(cx1)

	        CALL COPY_C_STRING(com1(1,i,j,k,l),
     .                             res(1,m10lox,m10loy,i2,m10lot))

	     i2 = i2 + 1

 1660	  CONTINUE
        ENDIF
	RETURN


* "TSEQUENCE" - reshape/flatten the input into a line in T
 1700	IF (typr .EQ. pfloat_words) THEN
	  i2 = m10lot
	  DO 1710 l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	   DO 1710 k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	    DO 1710 j = cx_lo_s2(cx1), cx_hi_s2(cx1)
	     DO 1710 i = cx_lo_s1(cx1), cx_hi_s1(cx1)

	     IF ( com1(1,i,j,k,l) .EQ. bad1 ) THEN
	        res(1,m10lox,m10loy,m10loz,i2) = bad_res
	     ELSE
	        res(1,m10lox,m10loy,m10loz,i2) = com1(1,i,j,k,l)
	     ENDIF
	     i2 = i2 + 1

 1710	  CONTINUE
        ELSE
	  i2 = m10lot
	  DO 1760 l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	   DO 1760 k = cx_lo_s3(cx1), cx_hi_s3(cx1)
	    DO 1760 j = cx_lo_s2(cx1), cx_hi_s2(cx1)
	     DO 1760 i = cx_lo_s1(cx1), cx_hi_s1(cx1)

	        CALL COPY_C_STRING(com1(1,i,j,k,l),
     .                             res(1,m10lox,m10loy,m10loz,i2))

	     i2 = i2 + 1

 1760	  CONTINUE
        ENDIF
	RETURN

* "SAMPLEI" - just copy arg 2 to result
 1800   CONTINUE        ! SAMPLEI
 1900   CONTINUE        ! SAMPLEJ
 2000   CONTINUE        ! SAMPLEK
 2100   CONTINUE        ! SAMPLEL
	IF (typr .EQ. pfloat_words) THEN
	  DO 1810 l = cx_lo_s4(cx2), cx_hi_s4(cx2)
	   DO 1810 k = cx_lo_s3(cx2), cx_hi_s3(cx2)
	    DO 1810 j = cx_lo_s2(cx2), cx_hi_s2(cx2)
	     DO 1810 i = cx_lo_s1(cx2), cx_hi_s1(cx2)

	     IF ( com2(1,i,j,k,l) .EQ. bad2 ) THEN
	        res(1,i,j,k,l) = bad_res
	     ELSE
	        res(1,i,j,k,l) = com2(1,i,j,k,l)
	     ENDIF

 1810	  CONTINUE

        ELSE

	  DO 1860 l = cx_lo_s4(cx2), cx_hi_s4(cx2)
	   DO 1860 k = cx_lo_s3(cx2), cx_hi_s3(cx2)
	    DO 1860 j = cx_lo_s2(cx2), cx_hi_s2(cx2)
	     DO 1860 i = cx_lo_s1(cx2), cx_hi_s1(cx2)

	     CALL COPY_C_STRING(com2(1,i,j,k,l), res(1,i,j,k,l))

 1860	  CONTINUE
        ENDIF
	RETURN

* SPAWN function - perform a Unix system command
 2200	CONTINUE
* ... get the command string to be executed
          IF (CGRID_SIZE(cx1) .NE. 1)  CALL ERRMSG(ferr_invalid_command,
     .          status, 'SPAWN can only execute a single command',*5000)
	  buff = GET_STRING_ELEMENT(
     .                  m1lox,m1hix,m1loy,m1hiy,m1loz,m1hiz,m1lot,m1hit,
     .			com1,m1lox,m1loy,m1loz,m1lot,max_cmnd_len,slen )
          IF (slen .EQ. max_cmnd_len) CALL ERRMSG(ferr_invalid_command,
     .          status, 'SPAWN command too long -- 500 char max',*5000)
* ... execute the command and store the results
          CALL STORE_SYS_STRINGS(buff,  res_cx, mres, 1, nnew, status)
          RETURN

* STRCMP function
 2300	CONTINUE
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

	     CALL C_STRCMP(com1(1,i1,j1,k1,l1),
     .                     com2(1,i2,j2,k2,l2), ii)
             res(1,i,j,k,l) = FLOAT(ii)

           ENDDO
          ENDDO
         ENDDO
        ENDDO
        RETURN

* *kob* 3/02 STRLEN function
 2400	CONTINUE
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

	     CALL C_STRLEN(com1(1,i1,j1,k1,l1),
     .                     ii)
             res(1,i,j,k,l) = ii

           ENDDO
          ENDDO
         ENDDO
        ENDDO
        RETURN


* *kob* 3/02 UPCASE function
 2500	CONTINUE
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

	     CALL C_UPCASE(com1(1,i1,j1,k1,l1),
     .          res(1,i1,j1,k1,l1))

           ENDDO
          ENDDO
         ENDDO
        ENDDO
        RETURN

* *kob* 3/02 STRINDEX function
 2600	CONTINUE
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

	     CALL C_STRINDEX(com1(1,i1,j1,k1,l1),
     .                       com2(1,i2,j2,k2,l2), ii)
             res(1,i,j,k,l) = FLOAT(ii)

           ENDDO
          ENDDO
         ENDDO
        ENDDO
        RETURN

* *kob* 3/02 STRRINDEX function
 2700	CONTINUE
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

	     CALL C_STRLEN(com2(1,i1,j1,k1,l1),
     .                     ii)
             IF (ii .LE. 0) THEN
	        CALL C_STRLEN(com1(1,i1,j1,k1,l1),
     .                     ii)
                res(1,i,j,k,l) = float(ii)
             ELSE
	        CALL C_STRRINDEX(com1(1,i1,j1,k1,l1),
     .                        com2(1,i2,j2,k2,l2), ii)
                res(1,i,j,k,l) = FLOAT(ii)
             ENDIF
 
           ENDDO
          ENDDO
         ENDDO
        ENDDO
        RETURN


* *kob* 3/02 DNCASE function
 2800	CONTINUE
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

	     CALL C_DNCASE(com1(1,i1,j1,k1,l1),
     .                     res(1,i,j,k,l))

           ENDDO
          ENDDO
         ENDDO
        ENDDO
        RETURN

 2900	CONTINUE
* *kob* STRCAT function 3/02
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

	     CALL C_STRCAT(com1(1,i1,j1,k1,l1),
     .                     com2(1,i2,j2,k2,l2),
     .                      res(1,i,j,k,l))

           ENDDO
          ENDDO
         ENDDO
        ENDDO
        RETURN

 3000	CONTINUE
* *kob* SUBSTRING function 3/02
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

	     CALL C_SUBSTR(com1(1,i1,j1,k1,l1),
     .                     com2(1,i2,j2,k2,l2), 
     .                     com3(1,i3,j3,k3,l3),
     .                     res(1,i,j,k,l))

           ENDDO
          ENDDO
         ENDDO
        ENDDO
        RETURN

* *kob* 5/02 STRFLOAT function
 3100	CONTINUE
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

	     CALL C_STRFLOAT(com1(1,i1,j1,k1,l1),
     .                       res(1,i,j,k,l))

           ENDDO
          ENDDO
         ENDDO
        ENDDO
        RETURN

 3200   CONTINUE
 3300   CONTINUE
 3400   CONTINUE
 3500   CONTINUE

	STOP 'GC FUNCTION NOT IMPLEMENTED'


* error exits
 5000	RETURN
	END
