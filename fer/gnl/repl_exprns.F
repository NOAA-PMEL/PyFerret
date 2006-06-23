	SUBROUTINE REPL_EXPRNS( memory, cmnd, lencmnd, cmnd_num,
     .				istart, max_check,
     .				digits, did_sub, status  )

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
*
* parse the command line replacing the first expression that is enclosed
* between grave accents with text strings representing their values
* this routine will be called once for each such expression so that the
* entire command line can be re-parsed between calls.  This makes it possible
* to build the command qualifier information from fixed limits and grave
* accented expressions working from left to right

* USAGE EXAMPLE:
*	CONTOUR/Z=`TEMP[Z=0:1000@LOC:15]` SALINITY*`some_factor`

* NOTE (3/00) THE POTENTIAL FOR WRONG RESULTS WITH OPERATIONS SUCH AS MATRIX
* TRANSPOSE - ALL "IMPOSED" AXES PRESENT THE RISK OF WRONG RESULTS.
* THIS COULD BE FIXED BY FORCING FULL EVALUATION OF EVERY EXPRESSION THAT
* CONTAINS A GC FUNCTION WITH AN IMPOSED AXIS. (For now we leave this
* bug in favor of the performance increase ... important in automated scripts
* that want to test the size of a result before computing it.


* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V400 3/95
* V411: 9/8/95 - serious bug fixed:  use CREATE_TEMP_MEM_VAR and COPY_GRID
*		so that the result scalar is picked correctly from within
*		the body of a larger mvar (instead of assuming that it is
*		always element (1,1,1,1) of the mvar)
* V420: 11/95 - allow grave accents to be "escaped" by '\'
* 	 1/96 - bug fix: special check if grave accents enclose white space
*	 2/96 - allow digits control with `expr,PREC=n`
*		and control of "bad string" with "BAD=xx.xx"
*	 3/96 - added RETURN=shape, isize,istart,iend, etc. for j,k, and l
* Linux Port - 5/97 *kob*
*	     - added ifdef check for double slash because f90/linux didn't
*	       need two of them together for escapes
* V510: 9/99 *sh* - added RETURN=XSTART (X-Y-Z-T, START-END-UNITS))
*	3/00 *sh* - RETURN= options computed without evaluating the expression
*		  - added RETURN=SIZE
* V522: 6/00 *sh* - added RETURN = XAXIS, YAXIS, ZAXIS, TAXIS
* V530: *sh* 9/00 - added support for string variables
*       *sh* 2/01 - added FORMAT = I4.4, etc. (NOT IMPLEMENTED)
*	          - added WIDTH=width and ZWIDTH=width
*	*sh* 3/01 - added RETURN=dset, dsetnum, dsetpath
* V531: *sh* 4/01 - added RETURN=bad
* V532: *sh* 5/01 - mod to EVAL_CONTEXT forced change here  
* *kob* 2/03 - replace call to intrinsic func CHAR with its octal
*              constant equivalent - needed for g77 port
*            - Need to use risc_buff for arbitrary concatenations
* v552: *acm* 4/03 - Issue a complete error message for wrong arguments
*                    to RETURN=
* v552: *acm* 4/03 - Add RETURN=dsettitle
*       *acm* 4/03 - Add RETURN=nc_scale, nc_off[set], user_scale, user_off[set]
* V570  *acm* 6/04 - Add RETURN=CALENDAR
* V580  *acm* 8/04 - Increase the length of string repl from 128 to 2048: If using a
*                    long string, e.g. in a label, this string can be very long. (bug 956)
* V581*acm*  6/05 - For fix to bug 1271, add flag line_shift_origin
*            and restore original t0 on output
* V600 *acm* 3/06  fixing bugs 439&1390, pass apply_cx to get_new_cx to say whether
*                  to apply command context to grave-accent expressions.
* V600 *acm* 6/05 - Attribute control, RETURN= attribute information.
* V600  4/06 *acm* - additional declarations found by Solaris compiler
* V600  6/06 *acm* - additional declarations found by Solaris compiler

        include 'netcdf.decl'
#include "netcdf.inc"
        include 'ferret.parm'
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xrisc.cmn'
	include 'xcontext.cmn'
	include 'xtext_info.cmn'
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	include 'xdset_info.cmn_text'
	external xdset_info_data
        include 'xdset_info.cd_equiv'

* calling argument declarations:
	LOGICAL	  did_sub, info_query, itsa_subsc, apply_cx
	INTEGER	  cmnd_num, lencmnd, istart, max_check, digits, status
	REAL	  memory( mem_blk_size, max_mem_blks )
	CHARACTER cmnd*(*)

* local parameter declarations
	INTEGER	   max_len
	PARAMETER (max_len = 2048)
!	INTEGER	   fmt_lenH
!	PARAMETER (fmt_len = 64 )

* internal variable declarations:
	LOGICAL   need_formatting, zero_fill, coordvar,
     .            do_err
	INTEGER   CGRID_SIZE, CGRID_AXIS, TM_LENSTR1, CX_DIM_LEN,
     .		  FIND_DSET_NUMBER, str_len, lenbuff, iend, itmp, axis,
     .		  mr, cx, mr_temp, i, i2, use_digits, true_end,
     .		  qual_end, brkt, idim, use_width, dset, varid, 
     .            vartype, nvdims, vdims(8), nvatts, ist, ind,
     .            attype, attlen, inxt, attoutflag,outflag, fertype, 
     .            attid, ieq
	REAL	  result, val
	REAL*8 TM_WORLD, ww
	CHARACTER TM_FMT*16, LEFINT*16, VAR_TITLE*(max_len),
     .		  VAR_UNITS*(max_len), GET_STRING_ELEMENT*(max_len),
     .		  repl*(max_len), bad_str*(max_len), return_errmsg*256,
     .            buff*128, varname*512, attname*128, aname*128
!	CHARACTER fmt*(max_len)
        CHARACTER*6 typstring(0:6)

* local parameter declarations
	CHARACTER	tab*1
#ifdef NO_INTRINSIC_IN_PARAMETER
	PARAMETER     ( tab = o'011' )
#else
	PARAMETER     ( tab = CHAR(9))
#endif

        DATA typstring/
     .  'Notype', 'BYTE', 'CHAR', 'SHORT', 'INT4', 'FLOAT', 'DOUBLE' /

* initialize
	did_sub = .FALSE.    	! any substitutions done?
	lenbuff = LEN( cmnd )
	use_digits = digits
	bad_str = 'bad'
!	fmt = ' '
	use_width = 0
	zero_fill = .FALSE.
	info_query = .FALSE.
        coordvar = .FALSE.
        apply_cx = .FALSE.

* do_err: issue error message if attrib does not exist on varname.attname
        do_err = .TRUE.

* begin the search/replace at the point specified in the call.  (This is done
* to facilitate double grave accents condensed into single accents so they
* can be passed to the command line in SPAWN commands)
* locate the start of the next grave accent pair
 10	IF ( istart .GE. lencmnd ) RETURN
	itmp = INDEX(cmnd(istart:),'`')
        IF ( itmp .EQ. 0 ) RETURN
	istart = istart + itmp
#ifdef NO_DOUBLE_ESCAPE_SLASH
	IF ( istart .GT. 2 ) THEN
	   IF (cmnd(istart-2:istart-2) .EQ. '\') GOTO 10  
	ENDIF
#else
	IF ( istart .GT. 2 ) THEN
	   IF (cmnd(istart-2:istart-2) .EQ. '\\') GOTO 10  
	ENDIF
#endif


* locate the end of the next grave accent pair
* Note: no check for backslash-escaped accent here because properly paired
*    accents only need to be checked at start of accent pair
	itmp = INDEX(cmnd(istart:),'`')
	IF ( itmp .EQ. 0 ) GOTO 5100		! unpaired grave accent
	IF ( itmp .EQ. 1 ) THEN
* ... replace double grave accents with single
	   cmnd = cmnd(:istart-1)//cmnd(istart+1:)
	   istart = istart + 1			! first char following accent
	   lencmnd = lencmnd - 1
	   GOTO 1000
	ENDIF
	iend = istart + itmp - 2
	true_end = iend

* do not perform substitutions beyond max_check characters
	IF ( istart .GE. max_check ) RETURN

* ====== 2/96
* Parse ",PRECISION=n" and/or ",BAD=xx.xx" at the end of the expression
*  (may be abbreviated to "P=n" and "B=xx.xx")
* 3/96: added RETURN= (abbreviated R=)
* ... hunt for the first equal sign that is not in square brackets
	brkt = 0
	DO 90 i = istart, iend
	   IF (cmnd(i:i).EQ.'[') THEN
	      brkt = brkt + 1
	   ELSEIF (cmnd(i:i).EQ.']') THEN
	      brkt = brkt - 1
	   ELSEIF (cmnd(i:i).EQ.'=') THEN
	      IF ( brkt .EQ. 0 ) GOTO 98
	   ENDIF
 90	CONTINUE
	GOTO 149	! nope - no equal sign
* ... see if the first part of the expression is a d=dsetnum or d=dsetname
 98	inxt = i
        ieq = i
        IF (cmnd(i-1:i-1).EQ.'d') THEN
           inxt = i+1
           DO 91 i = inxt,iend
              IF ( cmnd(i:i) .EQ. ',' ) GOTO 94
 91        CONTINUE
 94        dset = FIND_DSET_NUMBER(cmnd(inxt:i-1))
       ENDIF

* ... hunt for the next equal sign 
	DO 93 i = ieq, iend
           IF (cmnd(i:i).EQ.'=') GOTO 99
 93	CONTINUE
	GOTO 149	! nope - no equal sign

* ... locate the comma following the mathematical expression
  99	DO 100 i = i, istart, -1
 100	IF ( cmnd(i:i) .EQ. ',' ) GOTO 101
	GOTO 5500			! no comma ???
 101	iend = i - 1
* ... now decode the name/value pairs following the comma
 110	i = i + 1	   ! advance to character following the comma
* ... locate the end of the name/value pair string
	DO 120 i2 = i, true_end   
	   IF ( cmnd(i2:i2) .EQ. ',' ) THEN
	      qual_end = i2 - 1	
	      GOTO 121
	   ENDIF
 120	CONTINUE
	qual_end = true_end
 121	CONTINUE
* ... locate the keyword start and decode the keyword/value pair
	DO 130 i = i, qual_end   
	   IF (cmnd(i:i).EQ.' ' .OR. cmnd(i:i).EQ.tab) THEN
	      CONTINUE
	   ELSEIF (cmnd(i:i).EQ.'p' .OR. cmnd(i:i).EQ.'P') THEN
* ... decode PRECISION = #_digits
	      CALL EQUAL_VAL(cmnd(i:qual_end), result, status)
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      IF (ABS(result) .GT. 10.) GOTO 5510 	         
	      use_digits = NINT(result)
	      GOTO 131
	   ELSEIF (cmnd(i:i).EQ.'b' .OR. cmnd(i:i).EQ.'B') THEN
* ... decode BAD = string
	      CALL EQUAL_STR_LC(cmnd(i:qual_end), bad_str, status)
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      GOTO 131
	   ELSEIF (cmnd(i:i).EQ.'r' .OR. cmnd(i:i).EQ.'R') THEN
* ... decode RETURN = string
	      CALL EQUAL_STRING(cmnd(i:qual_end), repl, status) !repl is buffer
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      info_query = .TRUE.
	      GOTO 131
	   ELSEIF (cmnd(i:i).EQ.'w' .OR. cmnd(i:i).EQ.'W') THEN
* ... decode width = width (blank-filled width)
	      CALL EQUAL_VAL(cmnd(i:qual_end), val, status)
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      IF (val.GT.0 .AND. val.LT.max_len) use_width = INT(val)
	      GOTO 131
	   ELSEIF (cmnd(i:i).EQ.'z' .OR. cmnd(i:i).EQ.'Z') THEN
* ... decode zwidth = width (zero-filled width)
	      CALL EQUAL_VAL(cmnd(i:qual_end), val, status)
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      IF (val.GT.0 .AND. val.LT.max_len) use_width = INT(val)
	      zero_fill = .TRUE.
	      GOTO 131
! unimplemented code to allow user to control format
!	   ELSEIF (cmnd(i:i).EQ.'f' .OR. cmnd(i:i).EQ.'F') THEN
!* ... decode FORMAT = string
!	      CALL EQUAL_STRING(cmnd(i:qual_end), fmt, status)
!	      IF ( status .NE. ferr_ok ) GOTO 5000
!	      GOTO 131
	   ELSE
	      GOTO 5500
	   ENDIF
 130	CONTINUE
 131	i = qual_end + 1	! comma or beyond end
* ... go back for another keyword?
	IF ( i .LT. true_end ) GOTO 110
* ======

* is the expression just white space?	(1/96 ... quick fix with GOTO's)
 149	DO 150 i = istart, iend
 150	IF ( cmnd(i:i) .NE. ' ' .AND. cmnd(i:i) .NE. tab ) GOTO 160
* ... weird - they gave us pure white space - no numerical result possible
	repl = bad_str
	str_len = 3
	GOTO 500

* is this a single number request or a RETURN= query?
 160	IF ( info_query ) THEN
* ... evaluate the context without computing the expression
*	this is a major performance consideration, but not 100% reliable
	   CALL GET_NEW_CX( cx_last, cx_cmnd, apply_cx, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
           ist = istart
           ind = iend 

	   IF (cmnd(istart:istart) .EQ. '(') THEN
              dset = cx_data_set(cx_cmnd)   ! initial value to try; will return dset
              CALL ISIT_COORD_VAR ( cmnd(istart:iend), dset, 
     .            buff, coordvar, status )
              IF ( status .NE. ferr_ok ) GOTO 5000

           ELSE
* ... context meaningless for d=1,return=varnames
              IF (cmnd(istart:istart+1) .NE. 'd=') THEN
                 CALL EVAL_CONTEXT ( cx_cmnd, cmnd(istart:iend), status )
                 IF ( status .NE. ferr_ok ) GOTO 5000

                 cx = is_cx(1)	! always comes back as the first context

* ... if the result has any axes with unknown limits then we have to do a
*	full evaluation. This happens (only?) from grid-changing fcns that
*	return ABSTRACT axes, on which limits may depend on data values
* NOTE THE POTENTIAL FOR WRONG RESULTS WITH OPERATIONS SUCH AS MATRIX
* TRANSPOSE - ALL "IMPOSED" AXES PRESENT THE RISK OF WRONG RESULTS

                 DO 190 idim = 1, 4
	            IF ( cx_hi_ww(idim, cx) .EQ. unspecified_val8
     .	           .AND. CGRID_AXIS(idim, cx) .NE. mnormal ) THEN
	               CALL EVAL_EXPR ( memory, cx_last,
     .		                     cmnd(istart:iend), apply_cx, status )
	               IF ( status .NE. ferr_ok ) GOTO 5000
	               cx = is_cx(1)	! always comes back as the first context
	              GOTO 200
	            ENDIF
 190	         CONTINUE
              ENDIF  ! `d=1,return=varnames`

	   ENDIF
* ... RETURN= information request: "repl" contains the keyword
*     Add to this error string when adding new arguments:

           WRITE (return_errmsg,*)  
     .       'use RETURN= shape,size,grid,title,bad,t0,units,dset, '//pCR//
     .	     '  dsetnum,dsetpath,dsettitle,*size,*start,*end,'//pCR//
     .       '  *units,*axis,nc_scale,nc_offset,user_scale,user_offset,'//pCR//
     .       '  calendar,dtype'

* ... RETURN=SHAPE

 200	   IF ( repl .EQ. "SHAPE" ) THEN
	      repl = ' '
	      str_len = 0
	      DO 210 idim = 1, 4
	         IF ( CX_DIM_LEN( idim, cx ) .GT. 1 ) THEN
	            str_len = str_len + 1
	            repl(str_len:str_len) = ww_dim_name(idim)
	         ENDIF
 210	      CONTINUE
	      IF (str_len .EQ. 0) THEN
	         repl = "POINT"
	         str_len = 5
	      ENDIF

* ... RETURN=T0
	   ELSEIF ( repl .EQ. "T0" ) THEN
	      repl = ' '	! default
	      axis = CGRID_AXIS ( T_dim, cx )
	      IF ( axis.NE.mnormal) THEN
	          IF (line_direction(axis) .EQ. 'TI') THEN
    		      repl = line_T0(axis)
                      IF (line_shift_origin(axis)) 
     .                  repl =  '01-JAN-0001 00:00:00'
                  ENDIF
	      ENDIF
              
* ... RETURN=CALENDAR
	   ELSEIF ( repl(1:3) .EQ. "CAL" ) THEN
	      repl = ' '	! default
	      axis = CGRID_AXIS ( T_dim, cx )
	      IF ( axis.NE.mnormal) THEN
	          IF (line_direction(axis) .EQ. 'TI' .OR.
     .                line_direction(axis) .EQ. 'TT')
     .		      repl = line_cal_name(axis)
	      ENDIF

* ... RETURN=UNITS
	   ELSEIF ( repl(1:4) .EQ. "UNIT" ) THEN	! or "=UNITS"
	      repl = VAR_UNITS( cx )

* ... RETURN=TITLE
	   ELSEIF ( repl .EQ. "TITLE" ) THEN
	      repl = VAR_TITLE( cx )

* ... RETURN=GRID
	   ELSEIF ( repl .EQ. "GRID" ) THEN
	      repl = grid_name( cx_grid(cx) )

* ... RETURN=DSET, DSETNUM, DSETPATH, DSETTITLE
	   ELSEIF ( repl(1:4) .EQ. "DSET" ) THEN
	      i2 = cx_data_set(cx)
	      IF ( i2 .EQ. pdset_irrelevant
     .	     .OR.  i2 .EQ. unspecified_int4 ) THEN
	         repl = bad_str
	      ELSE
	         IF ( repl .EQ. "DSETNUM") THEN
	            repl = LEFINT(i2, str_len)
	         ELSEIF ( repl .EQ. "DSETPATH" ) THEN
	            CALL GET_DSET_NAME(i2, repl, str_len)
	         ELSEIF ( repl .EQ. "DSET" ) THEN
	            CALL GET_SHORT_DSET_NAME(i2, repl, str_len)
	         ELSEIF ( repl .EQ. "DSETTITLE" ) THEN
	            CALL GET_DSET_TITLE(i2, repl, str_len)
	         ELSE 
	            GOTO 5520
	         ENDIF
	      ENDIF

* ... RETURN=SIZE
	   ELSEIF ( repl .EQ. "SIZE"  .AND. 
     .              INDEX(cmnd(istart:iend),'.') .EQ. 0) THEN
	      i2 = CGRID_SIZE ( cx )
	      repl = LEFINT(i2, str_len)

* ... RETURN=BAD
	   ELSEIF ( repl .EQ. "BAD" ) THEN
	      result = cx_bad_data ( cx )
	      repl = TM_FMT( result, use_digits, max_len, str_len )

* ... RETURN=nc_scale, nc_offset
	   ELSEIF ( repl(1:3) .EQ. "NC_" ) THEN
              IF ( repl .EQ. "NC_SCALE" ) THEN
	         CALL GET_NC_SCALE (cx, result)
	      ELSEIF ( repl(1:6) .EQ. "NC_OFF" ) THEN
	         CALL GET_NC_OFFSET (cx, result)
	      ELSE 
	         GOTO 5520
	      ENDIF
	      repl = TM_FMT( result, use_digits, max_len, str_len )

* ... RETURN=user_scale, user_offset
	   ELSEIF ( repl(1:5) .EQ. "USER_" ) THEN
              IF ( repl .EQ. "USER_SCALE" ) THEN
	         CALL GET_USER_SCALE (cx, result)
	      ELSEIF ( repl(1:8) .EQ. "USER_OFF" ) THEN
                 CALL GET_USER_OFFSET (cx, result)
	      ELSE 
	         GOTO 5520
	      ENDIF
	      repl = TM_FMT( result, use_digits, max_len, str_len )

* ... varname ,RETURN=dtype
	   ELSEIF ( repl .EQ. "DTYPE" .AND. 
     .              INDEX(cmnd(istart:iend),'.') .EQ. 0) THEN
              dset = cx_data_set(cx)
              IF (dset .EQ. pdset_irrelevant .OR. 
     .            dset .EQ. unspecified_int4) dset = -1  ! try user vars

              varid = 0
              vartype = 0

              CALL CD_GET_VAR_ID(dset, cmnd(istart:iend), varid, status)

              IF (status .EQ. ferr_ok)  
     .          CALL CD_GET_VAR_INFO (dset, varid, cmnd(istart:iend), 
     .                              vartype, nvdims, vdims, nvatts,
     .                              coordvar, outflag, status )

              IF (vartype.EQ.0 ) THEN  

*  evaluate expr to get type and translate ferret type to netcdf

	         CALL EVAL_EXPR ( memory, cx_last,
     .		                     cmnd(istart:iend), apply_cx, status )
	         IF ( status .NE. ferr_ok ) GOTO 5000
                 fertype = cx_type(cx)

                 IF (fertype .EQ. ptype_unknown) THEN
                    vartype = 0  ! NAT = 'Not A Type' (c.f. NaN) 
                 ELSE IF (fertype .EQ. ptype_float) THEN
                    vartype = 5    ! NC_FLOAT
                 ELSE IF (fertype .EQ. ptype_int4) THEN
                    vartype = 4     ! NC_INT
                 ELSE IF (fertype .EQ. ptype_int2) THEN
                    vartype = 3     ! NC_SHORT
                 ELSE IF (fertype .EQ. ptype_int1) THEN
                    vartype = 1     ! NC_BYTE
                 ELSE IF (fertype .EQ. ptype_char) THEN
                    vartype = 2     ! NC_CHAR
                 ELSE IF (fertype .EQ. ptype_string) THEN
                    vartype = 2   ! NC_CHAR
                 ELSE IF (fertype .EQ. ptype_double) THEN
                    vartype = 6   ! NC_DOUBLE
                 ELSE
                    vartype = 0
                 ENDIF
              ENDIF
              repl = typstring(vartype)

* ... varname.atttnum,RETURN=dtype  or varname.attname, RETURN=dtype
	   ELSEIF ( repl .EQ. "DTYPE" .AND. 
     .              INDEX(cmnd(istart:iend),'.') .GT. 0) THEN
              attype = 0
              dset = cx_data_set(cx)
              IF (cx_data_set(cx) .EQ. pdset_irrelevant) dset = -1  ! user vars
              CALL BREAK_VARATTNAME( cmnd(istart:iend), dset, 
     .                    varname, attname, varid, do_err, status )
              CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, 
     .              status)
              IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .              attid, aname, attype, attlen, attoutflag, status )

              IF (status .EQ. ferr_ok) THEN
                 repl = typstring(attype)
	      ELSE 
	         GOTO 5520
              ENDIF

* ... varname.atttnum,RETURN=size  or varname.attname, RETURN=size
	   ELSEIF ( repl .EQ. "SIZE" .AND. 
     .              INDEX(cmnd(istart:iend),'.') .GT. 0) THEN
              IF (.NOT. coordvar) dset = cx_data_set(cx)
              IF (dset .GE. 1) THEN
                 IF (coordvar) THEN

                    IF (INDEX(buff,'.') .GT. 0)
     .              CALL BREAK_VARATTNAME( buff, dset, varname, 
     .                              attname,  varid, do_err, status )

                 ELSE
                    CALL CD_GET_VAR_ID (dset, cmnd(istart:iend), varid,
     .                  status)
                    
                    IF (status .NE. ferr_ok .AND. 
     .                  INDEX(cmnd(istart:iend),'.') .GT. 0)
     .              CALL BREAK_VARATTNAME( cmnd(istart:iend), dset,
     .                         varname, attname, varid, do_err, status )

                 ENDIF
                 
                 CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, 
     .               status)
                 IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid,
     .               attid, aname,  attype, attlen, attoutflag, status )

                 IF (status .EQ. ferr_ok) THEN
                    result = attlen
                    repl = TM_FMT( result, use_digits, max_len, str_len )
	         ELSE 
	            GOTO 5520
                 ENDIF
              ENDIF

* ... RETURN=*SIZE, *START, *END, *UNITS, *AXIS
	   ELSE	
* ... axis specific requests - must start with i,j,k,l or x,y,z,t
	      DO 220 idim = 1, 4
	         IF ( repl(1:1) .EQ. ss_dim_name(idim) ) THEN
	            itsa_subsc = .TRUE.
	            GOTO 221
	         ELSEIF ( repl(1:1) .EQ. ww_dim_name(idim) ) THEN
	            itsa_subsc = .FALSE.
	            GOTO 221
	         ENDIF
 220	      CONTINUE
	      GOTO 5520
 221	      need_formatting = .TRUE.
	      IF ( repl(2:5) .EQ. "SIZE" ) THEN
	         IF (.NOT.itsa_subsc) GOTO 5520
	         i2 = CX_DIM_LEN(idim, cx)
	      ELSEIF ( repl(2:6) .EQ. "START" ) THEN
	         i2 = cx_lo_ss(cx,idim)
	      ELSEIF ( repl(2:4) .EQ. "END" ) THEN
	         i2 = cx_hi_ss(cx,idim)
	      ELSEIF ( repl(2:5) .EQ. "AXIS" ) THEN
	         axis = CGRID_AXIS ( idim, cx )
	         IF (axis.EQ.mnormal .OR. axis.EQ.munknown) THEN
	           repl = 'NORMAL'
	         ELSE
	            repl = line_name( axis )
	         ENDIF
	         need_formatting = .FALSE.
	      ELSEIF ( repl(2:5) .EQ. "UNIT" ) THEN
	         axis = CGRID_AXIS ( idim, cx )
	         IF (axis.EQ.mnormal .OR. axis.EQ.munknown) THEN
	           repl = ' '
	         ELSE
	           repl = line_units( axis )
	         ENDIF
	         need_formatting = .FALSE.
	      ELSE
	         GOTO 5520
	      ENDIF
	      IF ( need_formatting ) THEN
	         IF ( itsa_subsc) THEN
	            IF (i2 .EQ. unspecified_int4) i2 = 0
	            repl = LEFINT(i2, str_len)
	         ELSE
	            ww = TM_WORLD(i2, cx_grid(cx), idim, box_middle)
! unimplemented code to allow user to control format
!	            IF ( fmt .NE. ' ' ) THEN
!*    ... use user-specified time format
!	              CALL TRANSLATE_FORMATTED( ww, idim, cx_grid(cx),
!     .					        fmt, bad_str, repl )
!	            ELSE
*   ... default formatting for world coordinates
	              CALL TRANSLATE_TO_WORLD( ww, idim, cx_grid(cx),
     .				             use_digits, repl )
!	            ENDIF
	         ENDIF
	      ENDIF
	   ENDIF
	   str_len = TM_LENSTR1(repl)

	ELSE
* single number answer desired
* ... evaluate the expression
	   CALL EVAL_EXPR ( memory, cx_last, cmnd(istart:iend), 
     .        .TRUE., status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   cx = is_cx(1)	! always comes back as the first context
	   mr = is_mr(1)

* ... make sure the result is a single, valid value
	   IF ( CGRID_SIZE(cx) .NE. 1 ) GOTO 5200	! result not a scalar

* ... extract the result scalar from the (possibly larger) memory variable
	   CALL CREATE_TEMP_MEM_VAR( cx, mr_temp, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   CALL COPY_GRID( memory(1, mr_blk1(mr)     ), mr,
     .                     memory(1, mr_blk1(mr_temp)), mr_temp )

	   IF (mr_type(mr_temp) .EQ. ptype_string ) THEN

* ... get one string
	     repl =  GET_STRING_ELEMENT(
     .				mr_lo_s1(mr_temp), mr_hi_s1(mr_temp),
     .				mr_lo_s2(mr_temp), mr_hi_s2(mr_temp),
     .				mr_lo_s3(mr_temp), mr_hi_s3(mr_temp),
     .				mr_lo_s4(mr_temp), mr_hi_s4(mr_temp),
     .				memory(1, mr_blk1(mr_temp)),
     .				cx_lo_s1(cx), cx_lo_s2(cx),
     .				cx_lo_s3(cx), cx_lo_s4(cx),
     .				max_len, str_len)
	   ELSE

* .... convert to a formatted result with requested digits
	     result =  memory(1, mr_blk1(mr_temp))
	     IF ( result .EQ. mr_bad_data(mr) ) THEN
	        repl = bad_str
	        str_len = TM_LENSTR1(bad_str)
	     ELSE
	        repl = TM_FMT( result, use_digits, max_len, str_len )
	     ENDIF
	   ENDIF
* ... clean up temporary variable
	   CALL DELETE_VARIABLE( mr_temp )
	ENDIF

* adjust the field width if requested by the user
 500	IF (use_width .GT. 0) THEN
	   IF (str_len .LT. use_width) THEN
	      i2 = use_width - str_len
	      DO i = use_width, i2+1, -1   ! shift characters to right
	        repl(i:i) = repl(i-i2:i-i2)
	      END DO
	      DO i = 1, i2
	        IF (zero_fill) THEN
	          repl(i:i) = '0'
	        ELSE
	          repl(i:i) = ' '
	        ENDIF
	      END DO
	   ENDIF
	   str_len = use_width
	ENDIF

* replace the expression text (incl grave accents) with the formatted string
	IF ( (istart-1)+str_len+(lencmnd-true_end-1) .GT. lenbuff ) THEN
           GOTO 5400					! too big to fit
        ELSE
	   IF ( iend+2 .LE. lencmnd ) THEN
	      risc_buff = cmnd(true_end+2:)
	   ELSE
	      risc_buff = ' '
	   ENDIF
           cmnd(istart-1:) = repl(:str_len) // risc_buff
           lencmnd = lencmnd - ((true_end-istart+3) - str_len)
        ENDIF

* successful translation of a grave-accented expression
 1000	did_sub = .TRUE.
	RETURN

* error exits
 5000   RETURN
 5100	risc_buff = cmnd(:lencmnd)
	CALL ERRMSG( ferr_syntax, status,
     .       'unpaired grave accents: '//risc_buff, *5000)
 5200	risc_buff = cmnd(istart:iend)
	CALL ERRMSG( ferr_invalid_command, status,
     .       'grave accent doesnt evaluate to scalar'//pCR
     .                                  //risc_buff, *5000)
 5400	risc_buff = cmnd(istart:iend)
	CALL ERRMSG( ferr_invalid_command, status,
     .       'grave accent evaluates to string too long'//pCR
     .					//risc_buff, *5000)
 5500	risc_buff = cmnd(:lencmnd)
	CALL ERRMSG( ferr_syntax, status,
     .       'options: "P=prec","B=bad","W=width","ZW=zero-width" or'
     .	     //' "R=return-item"'//pCR//risc_buff, *5000)
 5510	risc_buff = cmnd(i:qual_end)
	CALL ERRMSG( ferr_invalid_command, status,
     .       'maximum precision is 10 digits: '
     .					//risc_buff, *5000)
 5520   risc_buff = cmnd(i:qual_end)
	CALL ERRMSG( ferr_invalid_command, status,
     .     return_errmsg//pCR//risc_buff, *5000)
	END
