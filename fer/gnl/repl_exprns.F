	SUBROUTINE REPL_EXPRNS(  cmnd, lencmnd, cmnd_num,
     .				istart, max_check,
     .				ndigits, did_sub, status  )

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
* V601 10/06 *acm* - fix bug 1447, where say `var[d=fname.cdf],return=size` failed
*            parsing error, confusing the dot in the filename with `var.att,return=size`
* V602  4/07 *acm* - Add RETURN=xmod,tmod
* V615 11/08 *acm* - fixing bug 1523. If the variable contains a grid-changing 
*                    function, we will do a full evaluation to get its size
* V68  *acm* 1/12  ifdef double_p for double-precision ferret.
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* V684 *acm* 12/12 Add data type of DOUBLE for return=dtype
* V685 *sh*  11/13 Add return=status and return=ready
* V69+ *acm*  9/14 Ticket 2197. After return=status and return=ready need to unset the
*                  flag dependency_tree_mode to turn regular error reporting back on.
* V69+ *acm*  9/14 Ticket 981. If the variable is a constant-array, evaluate it to get
*                  the grid size rather than returning the nominal length.
* V695 *acm*  2/15 RETURN=F0 for calendar formatted F axis 
* V695+ *sh* 2/15 add support for _DQ_ style of quotations
* V694 *acm* 5/15 fix bug 2213
* V695 *acm* 6/15 Add return=defined (ticket 2293)
* v695 *acm* 9/15 Ticket 2313: For `return=`, if the return request is for an attribute 
*                 not related to the grid or other characteristics that need evaluation, 
*                 do not evaluate the expression.
* v695 *acm* 5/16 Ticket 2389. When the user gives `var,prec=xx`, return the acutal number
*                 of digits requested. Return exponential formats when that makes the
*                 result shorter. Changes are only for positive values of ,prec=
* v701 *acm* 9/16 Ticket 1269: For RETURN=shape, use the same logic as in SHOW GRID,
*                 for determining when a grid dimension is compressed by a transformation.
* V702 1/17 *sh* for trac enhancement #2369 -- dynamic memory management 
* V720 *acm* 5/17 Implement specific calls in place of CD_GET_VAR_INFO
* V73+ *acm* 12/17 Fix issue #1847 - value overflows integer on RETURN=size
* V74  *acm* 2/18 Issue # 1861, handling missing- and _FillValue with scale and offset
*                 change to `var,RETURN=bad`
* V74+ *acm*10/10 Partial fix to working with E contexts in dsg datasets.
* v745 *acm* 12/18  Issue 1909, option to write dates in ISO8601 format
* v7441*acm* 12/18 Issue 1910: Allow dates in commands to be in ISO 8601 form, yyyy-mm-dd
* v745* acm*  2/19 Issue 400:  If a user var is LET var = filevar[d=1] then return the
*                  correct info for `var,return=dset` and dsetnum and dsetpath.
* V745  4/19 *acm* ticket 1916, special handling getting context for RETURN= outputs
* V751  6/19 *acm* ticket 1929, don't chop off str_len when testing for reformatting at the end
* V7.63 9/20 *acm* add `var,RETURN=*coord` to get coordinate variables for DSG datasets

        IMPLICIT NONE
#include "netcdf.inc"
	include 'tmap_dims.parm'
        include 'ferret.parm'
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xrisc.cmn'
	include 'xcontext.cmn'
	include 'xtext_info.cmn'
	include 'xdependency_tree.cmn'
	include 'xtm_grid.cmn_text'
	include 'xdset_info.cmn_text'
	external xdset_info_data
        include 'xdset_info.cd_equiv'

* calling argument declarations:
	LOGICAL	  did_sub, info_query, itsa_subsc, apply_cx
	INTEGER	  cmnd_num, lencmnd, istart, max_check, ndigits, status
	CHARACTER cmnd*(*)

* local parameter declarations 
	INTEGER	   max_len
	PARAMETER (max_len = 2048)

* internal variable declarations:
	LOGICAL   TM_HAS_STRING, TM_DFPEQ, TM_ITSA_DSG, 
     .		  need_formatting, zero_fill, coordvar,
     .            do_err, do_eval, get_ready, was_dep, const_arr_var,
     .		  dqote, DQ, orig_name, uvgrid, compressed, grid_reset, is_dsg
	INTEGER   CGRID_SIZE, CGRID_AXIS, TM_LENSTR1, CX_DIM_LEN,
     .		  FIND_DSET_NUMBER, GET_ABSTRACT_LINE, TM_LENSTR,
     .		  TM_GET_GRIDNUM, GRID_FROM_NAME, TM_SUBSC, TM_GET_CALENDAR_ID,
     .		  TM_DSG_DSET_FROM_GRID,
     .		  str_len, lenbuff, iend, itmp, axis,
     .		  mr, cx, mr_temp, i, i1, i2, use_digits, true_end,
     .		  qual_end, brkt, idim, use_width, dset, varid,
     .            vartype, attype, attlen, inxt, attoutflag, fertype,
     .            attid, ieq, base_isp, base_cx, cat, var, uvar,
     .            i1ss, inss, grid, ucode, dotidx, shorterlen,
     .            hi_cx, item, itype, start, len_name, var_status, 
     .            siz, ifeature
	REAL	  result, val, aval
	REAL*8 TM_WORLD, TM_WW_AXLEN, ww
	CHARACTER TM_FMT*48, LEFINT*16, VAR_TITLE*(max_len),
     .		  VAR_UNITS*(max_len), GET_STRING_ELEMENT*(max_len), 
     .		  VAR_CODE*1024,
     .		  repl*(max_len), bad_str*(max_len), return_errmsg*256,
     .            buff*128, varname*512, attname*128, aname*128, name*128
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
	use_digits = ndigits
	bad_str = 'bad'
!	fmt = ' '
	use_width = 0
	zero_fill = .FALSE.
	info_query = .FALSE.
        coordvar = .FALSE.
        apply_cx = .FALSE.
	str_len = 0
	is_dsg = .FALSE.
	
	grid = unspecified_int4
	cx_only = .FALSE.   ! do merge_grid in mode for not evaluating full context

* do_err: issue error message if attrib does not exist on varname.attname
        do_err = .TRUE.

* begin the search/replace at the point specified in the call.  (This is done
* to facilitate  grave accents condensed into single accents so they
* can be passed to the command line in SPAWN commands)
* locate the start of the next grave accent pair
 10	IF ( istart .GE. lencmnd ) RETURN
	itmp = INDEX(cmnd(istart:),'`')
        IF ( itmp .EQ. 0 ) RETURN
	istart = istart + itmp
#ifdef NO_DOUBLE_ESCAPE_SLASH
	IF ( istart .GT. 2 ) THEN
	   IF (cmnd(istart-2:istart-2) .EQ. '\') GOTO 10  ! ' for emacs
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
* ... replace  grave accents with single
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

* ... see if the equals sign is just part of some string within the grave accents
*     This fixes bug 1469 (and its duplicate 1690)
* *sh* add support for _DQ_ style of quotations
	dqote = .FALSE. 
	DQ = .FALSE.
	DO i = istart, iend
	   IF (cmnd(i:i).EQ.'"' .AND. .NOT.DQ)  THEN
	      dqote = .NOT.dqote
	   ELSEIF (cmnd(i:i).EQ.'_'
     .       .AND. i+3 .LE. iend ) THEN
	      IF ( cmnd(i:i+3).EQ.'_DQ_' ) DQ = .NOT.DQ
	   ELSEIF (cmnd(i:i).EQ.'=' ) THEN
* ... does the "=" lie between quotation marks?
	      IF (dqote)  THEN
	         IF (INDEX(cmnd(i:iend), '"'   ) .GT. 0) GOTO 149
	      ELSEIF (DQ) THEN
	         IF (INDEX(cmnd(i:iend), '_DQ_') .GT. 0) GOTO 149
	      ENDIF
	   ENDIF
	ENDDO

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
	      IF (ABS(result) .GT. 16.) GOTO 5510 	         
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

* 11/2013
* return=ISREADY and return=STATUS are special.  They must function
*   correctly even in the face of unknown variables and unknown datasets.
*   So they must be handled before we attempt even to get the context for the
*   given expression.
	   IF (repl(1:4).EQ. 'STAT' .OR. repl(1:4) .EQ. 'ISRE') THEN
	      get_ready = repl(1:4) .EQ. 'ISRE'
	      CALL GET_DEPENDENCY_STATUS( cmnd(istart:iend),
     .					 repl, str_len, status) 
	      CALL EXIT_DEPENDENCY_MODE
              IF (status.NE. ferr_ok) GOTO 5050
	      IF (get_ready) THEN
*  return=READY gives merely a 1 or 0 
	         IF (    dependency_status .EQ. df_valid_var) THEN
	            repl = '1'
	         ELSE
	            repl = '0'
	         ENDIF
	      ENDIF
	      GOTO 500
	   ENDIF
	   
* 11/2013
* return=DEFINED needs only to check whether the user-variable is defined

* ... RETURN=DEFINED
	   IF ( repl(1:3) .EQ. "DEF" ) THEN
	      str_len = 1
	      repl = '0'	! default, pos=up, or not-depth.
	      CALL FIND_VAR_NAME ( pdset_irrelevant, cmnd(istart:iend), cat, var )
	      IF ( var .NE. munknown_var_name) repl = '1'
	      GOTO 500
	   ENDIF

* ... evaluate the context without computing the expression
*	this is a major performance consideration, but not 100% reliable

	   has_uvar_gc = .FALSE.
	   is_uvar_const = .FALSE.
	   CALL GET_NEW_CX( cx_last, cx_cmnd, apply_cx, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000

	   IF (cmnd(istart:istart) .EQ. '(') THEN
              dset = cx_data_set(cx_cmnd)   ! initial value to try; will return dset
              CALL ISIT_COORD_VAR ( cmnd(istart:iend), dset, 
     .            buff, coordvar, status )
              IF ( status .NE. ferr_ok ) GOTO 5000

           ELSE
* ... context meaningless for d=1,return=varnames
              IF (cmnd(istart:istart+1) .NE. 'd=') THEN

* are we going to need to unset things later? (context-setting for user-vars
* when doing shape and size returns.)

	         grid_reset = ( repl.EQ."SHAPE" .OR. 
     .              (repl.EQ."SIZE"  .AND. INDEX(cmnd(istart:iend),'.').EQ.0) )

	         IF ( repl(2:5).EQ."SIZE" ) THEN
	            DO idim = 1, nferdims
	               IF ( repl(1:1) .EQ. ss_dim_name(idim) ) THEN
		          grid_reset = .TRUE.
			  EXIT
	               ENDIF
		    ENDDO
	         ENDIF

	         IF ( repl(2:6).EQ."START" .OR. repl(2:4).EQ."END" ) THEN
	            DO idim = 1, nferdims
	               IF ( repl(1:1).EQ.ss_dim_name(idim).OR. 
     .                      repl(1:1).EQ.ww_dim_name(idim))  THEN
		          grid_reset = .TRUE.
			  EXIT
	               ENDIF
		    ENDDO
	         ENDIF

*  For results such as shape and size where we need to get the context
*  save a list of the uvars on the stack where the uvar grid gets set and
*  unset it later

		 IF (grid_reset) THEN
		    cx_only = .TRUE.

*  Allocate dynamic line memory for a list of uvars where uvar-grid info gets
*  set in this routine.
		    
		    CALL GET_LINE_DYNMEM(max_uvar, uvar_cx_only_lm, status)
		    uvar_cxo_len = 0

		    CALL DELETE_OLD_UGRIDS

		 ENDIF

		 CALL EVAL_CONTEXT ( cx_cmnd, cmnd(istart:iend), status )
                 IF ( status .NE. ferr_ok ) GOTO 5000

                 cx = is_cx(1)	! always comes back as the first context

		 cat = cx_category(cx)

* ... Returns that can be made without evaluating the expression. If success
*     is not verified, will continue and try again after evaluation.

	      IF (cx.EQ.unspecified_int4 .OR. cx.EQ.0) GOTO 4000 ! context not set

* ... RETURN=UNITS
              IF ( repl(1:4) .EQ. "UNIT" ) THEN	! or "=UNITS"
	         repl = VAR_UNITS( cx )
	         str_len = TM_LENSTR1(repl)
	         GOTO 500

* ... RETURN=TITLE
	      ELSEIF ( repl .EQ. "TITLE" ) THEN
	         repl = VAR_TITLE( cx )
	         str_len = TM_LENSTR1(repl)
	         GOTO 500
		  
* ... RETURN=BAD
	      ELSEIF ( repl .EQ. "BAD" ) THEN
	         result = cx_bad_data ( cx )

* if its a netcdf-type dset with packed data, may have swapped out the bad flag

	         i2 = cx_data_set(cx)
		 CALL GET_DSET_BAD( i2,  cmnd(istart:iend), result )

	         repl = TM_FMT( result, use_digits, 22, str_len )
	         GOTO 500
		 
* ... RETURN=T0
	      ELSEIF ( repl .EQ. "T0" .AND. cat.EQ.cat_file_var) THEN
	         repl = ' '	! default
	         idim = T_DIM
	         axis = CGRID_AXIS ( idim, cx )
	         IF ( axis.NE.mnormal) THEN
	            IF (line_direction(axis) .EQ. 'TI') THEN
    		       repl = line_T0(axis)
                       IF (line_shift_origin(axis)) 
     .                   repl =  '01-JAN-0001 00:00:00'
                    ELSE
		      GOTO 4000
		    ENDIF
	         ENDIF
	         str_len = TM_LENSTR1(repl)
	         CALL CHECK_DATE_REFORMAT (repl, str_len, 
     .              TM_GET_CALENDAR_ID ( line_cal_name(axis)), status)
	         GOTO 500

* ... RETURN=F0
	      ELSEIF ( repl .EQ. "F0" .AND. cat.EQ.cat_file_var) THEN
	         repl = ' '	! default
	         idim = F_DIM
	         axis = CGRID_AXIS ( idim, cx )
	         IF ( axis.NE.mnormal) THEN
	            IF (line_direction(axis) .EQ. 'FI') THEN
    		       repl = line_T0(axis)
                       IF (line_shift_origin(axis)) 
     .                  repl =  '01-JAN-0001 00:00:00'
                    ELSE
		      GOTO 4000
                    ENDIF
	         ENDIF
	         str_len = TM_LENSTR1(repl)
		 CALL CHECK_DATE_REFORMAT (repl, str_len, 
     .                       TM_GET_CALENDAR_ID ( line_cal_name(axis)))
	         GOTO 500
              
* ... RETURN=CALENDAR
	      ELSEIF ( repl(1:3) .EQ. "CAL" .AND. cat.EQ.cat_file_var) THEN
	         repl = ' '	! default
	         axis = CGRID_AXIS ( T_dim, cx )
	         IF ( axis.NE.mnormal) THEN
	               IF (line_direction(axis) .EQ. 'TI' .OR.
     .                  line_direction(axis) .EQ. 'TT')
     .		        repl = line_cal_name(axis)
	         ELSE
		      GOTO 4000
	         ENDIF
	         str_len = TM_LENSTR1(repl)
	         GOTO 500


* ... RETURN=DSET, DSETNUM, DSETPATH, DSETTITLE
	      ELSEIF ( repl(1:4) .EQ. "DSET" .AND. cat.EQ.cat_file_var) THEN
	         i2 = cx_data_set(cx)
	         IF ( i2 .EQ. pdset_irrelevant
     .	        .OR.  i2 .EQ. unspecified_int4 ) THEN
	            GOTO 4000
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
	               GOTO 4000
                    ENDIF
	         ENDIF
	         str_len = TM_LENSTR1(repl)
	         GOTO 500

* ... RETURN=nc_scale, nc_offset
	         ELSEIF ( repl(1:3) .EQ. "NC_" .AND. cat.EQ.cat_file_var) THEN
	            IF ( repl .EQ. "NC_SCALE" ) THEN
	               CALL GET_NC_SCALE (cx, result)
	            ELSEIF ( repl(1:6) .EQ. "NC_OFF" ) THEN
	               CALL GET_NC_OFFSET (cx, result)
                    ELSE 
	               GOTO 4000
	            ENDIF
	            repl = TM_FMT( result, use_digits, 22, str_len )
	            GOTO 500

* ... RETURN=user_scale, user_offset
	         ELSEIF ( repl(1:5) .EQ. "USER_" ) THEN
	            IF ( repl .EQ. "USER_SCALE" ) THEN
	               CALL GET_USER_SCALE (cx, result)
	            ELSEIF ( repl(1:8) .EQ. "USER_OFF" ) THEN
	               CALL GET_USER_OFFSET (cx, result)
	            ELSE 
	               GOTO 4000
	            ENDIF
	            repl = TM_FMT( result, use_digits, 22, str_len )
	            GOTO 500

* ... RETURN=dtype if it is available in the context at this point. For user-
*     defined variables the type is not yet determined.

	         ELSEIF ( repl .EQ. "DTYPE" .AND. cat.EQ.cat_file_var 
     .              .AND. INDEX(cmnd(istart:iend),'.') .EQ. 0) THEN
	            dset = cx_data_set(cx)
	            IF (dset .EQ. pdset_irrelevant .OR. 
     .               dset .EQ. unspecified_int4) dset = pdset_uvars  ! try user vars

	            varid = 0
	            vartype = 0

	            CALL CD_GET_VAR_ID(dset, cmnd(istart:iend), varid, status)

	            IF (status .EQ. ferr_ok)  
     .                CALL CD_GET_VAR_TYPE (dset, varid, cmnd(istart:iend), 
     .                              vartype, status )

                    IF (vartype.EQ.0 ) GOTO 4000
		    
	            IF (status .EQ. ferr_ok) THEN
                       repl = typstring(vartype)
	            ELSE 
	               GOTO 4000
	            ENDIF
	            str_len = TM_LENSTR1(repl)
	            GOTO 500

* ... varname.atttnum,RETURN=dtype  or varname.attname, RETURN=dtype

	         ELSEIF ( repl .EQ. "DTYPE" .AND. cat.EQ.cat_file_var
     .              .AND. INDEX(cmnd(istart:iend),'.') .GT. 0) THEN
	            attype = 0
	            dset = cx_data_set(cx)
	            IF (cx_data_set(cx) .EQ. pdset_irrelevant) dset = pdset_uvars  ! user vars
	               CALL BREAK_VARATTNAME( cmnd(istart:iend), dset, 
     .                    varname, attname, varid, do_err, status )
	               CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, 
     .                    status)
	               IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .                    attid, aname, attype, attlen, attoutflag, status )

	            IF (status .EQ. ferr_ok) THEN
                       repl = typstring(attype)
	            ELSE 
	               GOTO 4000
	            ENDIF
	            str_len = TM_LENSTR1(repl)
	            GOTO 500

* ... varname.atttnum,RETURN=size  or varname.attname, RETURN=size

	         ELSEIF ( repl .EQ. "SIZE" .AND. cat.EQ.cat_file_var
     .              .AND. INDEX(cmnd(istart:iend),'.') .GT. 0) THEN
	            IF (.NOT. coordvar) dset = cx_data_set(cx)
	            IF (dset .GE. 1) THEN
	               IF (coordvar) THEN

                          IF (INDEX(buff,'.') .GT. 0)
     .                    CALL BREAK_VARATTNAME( cmnd(istart:iend), dset, varname, 
     .                              attname,  varid, do_err, status )

	               ELSE
                          CALL CD_GET_VAR_ID (dset, cmnd(istart:iend), varid,
     .                        status)
                    
                          IF (status .NE. ferr_ok .AND. 
     .                     INDEX(cmnd(istart:iend),'.') .GT. 0) THEN
                             CALL BREAK_VARATTNAME( cmnd(istart:iend), dset,
     .                         varname, attname, varid, do_err, status )
                          ELSE
                             i2 = CGRID_SIZE ( cx )
                             repl = LEFINT(i2, str_len)
			     str_len = TM_LENSTR1(repl)
                             GOTO 500
                          ENDIF

	               ENDIF
                 
	               CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, 
     .                   status)
	               IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid,
     .                  attid, aname,  attype, attlen, attoutflag, status )

	               IF (status .EQ. ferr_ok) THEN
                          IF (attlen .GT. 0) THEN
                             result = attlen
                             repl = TM_FMT( result, use_digits, 22, str_len )
                          ENDIF
	               ELSE 
	                  GOTO 4000
                       ENDIF
                    ENDIF
	            str_len = TM_LENSTR1(repl)
	            GOTO 500

* ... RETURN=XMOD

	         ELSEIF ( repl .EQ. "XMOD" .AND. cat.EQ.cat_file_var) THEN
	            repl = ' '	! default
	            axis = CGRID_AXIS ( X_dim, cx )
	            IF ( axis.NE.mnormal) THEN
	               IF (line_modulo(axis)) THEN
                          result = line_modulo_len(axis)

! If axis has modulo=" ", and units are not degrees, mod len = axis len.
                          IF (result .EQ. 0) result = TM_WW_AXLEN(axis)
                          repl = TM_FMT( result, use_digits, 22, str_len)
		       ENDIF
	            ENDIF
	            str_len = TM_LENSTR1(repl)
	            GOTO 500

* ... RETURN=TMOD

	         ELSEIF ( repl .EQ. "TMOD" .AND. cat.EQ.cat_file_var) THEN
	            repl = ' '	! default
	            axis = CGRID_AXIS ( T_dim, cx )
	            IF ( axis.NE.mnormal) THEN
	               IF (line_modulo(axis)) THEN
                          result = line_modulo_len(axis)

! If axis has modulo=" ", mod len = axis len.
                          IF (result .EQ. 0) result = TM_WW_AXLEN(axis)
                          repl = TM_FMT( result, use_digits, 22, str_len)

                       ENDIF
	           ENDIF
	           str_len = TM_LENSTR1(repl)
	           GOTO 500

* ... RETURN=ISDEPTH

	         ELSEIF ( repl(1:3) .EQ. "ISD" .AND. cat.EQ.cat_file_var) THEN
	            axis = CGRID_AXIS ( Z_dim, cx )
	            IF ( axis.NE.mnormal) THEN
		       repl = '0'
	               IF (line_direction(axis) .EQ. "UD") repl = '1'
	            ELSE 
		       repl = '0'
	            ENDIF
	            str_len = TM_LENSTR1(repl)
	            GOTO 500

	         ENDIF  ! trying to return results based on context only

 4000        	CONTINUE

* ... if the result has any axes with unknown limits then we have to do a
*	full evaluation. This happens (only?) from grid-changing fcns that
*	return ABSTRACT axes, on which limits may depend on data values
* NOTE THE POTENTIAL FOR WRONG RESULTS WITH OPERATIONS SUCH AS MATRIX
* TRANSPOSE - ALL "IMPOSED" AXES PRESENT THE RISK OF WRONG RESULTS
* Just checking for the abstract axis to fix bug 2213. 

                 DO 190 idim = 1, nferdims
		    do_eval = ( cx_hi_ww(idim, cx) .EQ. unspecified_val8
     .	                        .AND. CGRID_AXIS(idim, cx) .NE. mnormal )
		    do_eval = (do_eval .OR. 
     .	                 CGRID_AXIS(idim, cx) .EQ. GET_ABSTRACT_LINE(idim) )
	            IF ( do_eval .OR. has_uvar_gc .OR. is_uvar_const) THEN
		       cx_only = .FALSE.
		       CALL EVAL_EXPR (  cx_last,
     .		                     cmnd(istart:iend), apply_cx, status )
	               IF ( status .NE. ferr_ok ) GOTO 5000
	               cx = is_cx(1)	! always comes back as the first context
		       cx_only = grid_reset
	               GOTO 200
	            ENDIF
 190	         CONTINUE
              ENDIF  ! `d=1,return=varnames`

* If it's a constant-array variable, we'd like to return its actual size, not the
* nominal length of the abstract axis. This should not be a performance problem.

              CALL ISIT_CONST_ARR_VAR ( cx, const_arr_var, status )
              IF (const_arr_var) CALL EVAL_EXPR (  cx_last,
     .		                     cmnd(istart:iend), apply_cx, status )

	   ENDIF
* ... RETURN= information request: "repl" contains the keyword
*     Add to this error string when adding new arguments:

           WRITE (return_errmsg,*)  
     .       'use RETURN= shape,size,grid,title,bad,t0,f0,units,dset, '//pCR//
     .	     '  dsetnum,dsetpath,dsettitle,*size,*start,*end,'//pCR//
     .       '  *units,*axis,*coord,nc_scale,nc_offset,user_scale,user_offset,'//pCR//
     .       '  calendar,dtype,xmod,tmod,status,isDepth,isReady,defined'
!     .       '  calendar,dtype,xmod,tmod,status,isDepth,isReady,isLoaded' ! w isLoaded

* ... RETURN=SHAPE

 200	   IF ( repl .EQ. "SHAPE" ) THEN
 
	      repl = ' '
	      str_len = 0

	      DO 210 idim = 1, nferdims
	         IF ( CX_DIM_LEN( idim, cx ) .GT. 1 ) THEN

		    IF (grid .EQ. unspecified_int4) 
     .                  grid = GRID_FROM_NAME ( cmnd(istart:iend), cx, status )
		    
* Getting details about the grid of a user-variable

		    i1 = TM_GET_GRIDNUM( cmnd(istart:iend) )
		    uvgrid = (i1 .EQ. unspecified_int4 )
		    uvar = -1
		    IF (uvgrid) uvar = is_uvar(isp)

		    compressed = .FALSE.
		    IF ( uvar_given(idim,uvar) .EQ. uvlim_gvn_xact)  THEN
		       IF (uvar_lo_ss(uvar, idim) .NE. unspecified_int4) THEN
		          i1ss = uvar_lo_ss(uvar, idim)
		          inss = uvar_hi_ss(uvar, idim)
		       ELSE
		          i1ss = TM_SUBSC ( uvar_lo_ww(idim, uvar), grid, idim, -1 )
		          inss = TM_SUBSC ( uvar_hi_ww(idim, uvar), grid, idim, -1 )
		       ENDIF
	               compressed = (i1ss .NE. inss) 
		    ENDIF

		    IF (.NOT.compressed) THEN
	               str_len = str_len + 1
	               repl(str_len:str_len) = ww_dim_name(idim)
	            ENDIF
	         ENDIF
 210	      CONTINUE
	      IF (str_len .EQ. 0) THEN
	         repl = "POINT"
	         str_len = 5
	      ENDIF

* ... RETURN=T0
	   ELSEIF ( repl .EQ. "T0" ) THEN
	      repl = ' '	! default
	      idim = T_DIM
	      axis = CGRID_AXIS ( idim, cx )
	      IF ( axis.NE.mnormal) THEN
	          IF (line_direction(axis) .EQ. 'TI') THEN
    		      repl = line_T0(axis)
                      IF (line_shift_origin(axis)) 
     .                  repl =  '01-JAN-0001 00:00:00'
                  ENDIF
	      ENDIF
	      str_len = TM_LENSTR1(repl)
	      CALL CHECK_DATE_REFORMAT (repl(:str_len*2), str_len, 
     .                    TM_GET_CALENDAR_ID ( line_cal_name(axis)))

* ... RETURN=F0
	   ELSEIF ( repl .EQ. "F0" ) THEN
	      repl = ' '	! default
	      idim = F_DIM
	      axis = CGRID_AXIS ( idim, cx )
	      IF ( axis.NE.mnormal) THEN
	          IF (line_direction(axis) .EQ. 'FI') THEN
    		      repl = line_T0(axis)
                      IF (line_shift_origin(axis)) 
     .                  repl =  '01-JAN-0001 00:00:00'
                  ENDIF
	      ENDIF
	      str_len = TM_LENSTR1(repl)
	      CALL CHECK_DATE_REFORMAT (repl, str_len, 
     .                    TM_GET_CALENDAR_ID ( line_cal_name(axis)))
              
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

******************-----------------
* borrow from is_dependencies to get uvar definition and find its dataset
* (is_dependencies has lifted this old-fashioned style code from elsewhere as well.)
* This makes just one try: see if it's a uservar and uvar_text clearly points 
* to a dataset and return that set.  More-complex expressions don't necessarily 
* have a clear dataset.

	      i2 = cx_data_set(cx)
	      IF (cat .EQ. cat_user_var) THEN
	         
	         uvar = cx_variable(cx)
	         item = 1

	         itype = uvar_item_type ( item, uvar )
	         start = uvar_item_start( item, uvar )
	         iend   = uvar_item_end  ( item, uvar )
	         IF ( itype .EQ. alg_variable    ) THEN

* variable: parse name, get expression context
	            name = uvar_text(uvar)(start:iend)
	            len_name = iend-start+1

	            CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	            IF ( status .NE. ferr_ok ) GOTO 5000
	            hi_cx = cx_stack_ptr
	            CALL GET_VAR_CONTEXT (  name(:len_name),
     .				cx,
     .				hi_cx,
     .				var_status )

		    IF (cx_data_set(hi_cx) .GT. pdset_irrelevant) i2 = cx_data_set(hi_cx)

	         ENDIF
	      ENDIF  ! if cat_user_var
	      
******************-----------------

* i2 contains the dataset number

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

* The above overflows integer storage? Compute as a real.
	      IF (i2 .LT. 0) THEN  
	         result = 1.
	         DO idim = 1, nferdims
	            i2 = CX_DIM_LEN(idim, cx)
		    result = result * i2
	         ENDDO
		 repl = TM_FMT( result, use_digits, 22, str_len )

	      ENDIF

* ... RETURN=BAD
	   ELSEIF ( repl .EQ. "BAD" ) THEN
	      result = cx_bad_data ( cx )
	      repl = TM_FMT( result, use_digits, 22, str_len )

* ... RETURN=nc_scale, nc_offset
	   ELSEIF ( repl(1:3) .EQ. "NC_" ) THEN
              IF ( repl .EQ. "NC_SCALE" ) THEN
	         CALL GET_NC_SCALE (cx, result)
	      ELSEIF ( repl(1:6) .EQ. "NC_OFF" ) THEN
	         CALL GET_NC_OFFSET (cx, result)
	      ELSE 
	         GOTO 5520
	      ENDIF
	      repl = TM_FMT( result, use_digits, 22, str_len )

* ... RETURN=user_scale, user_offset
	   ELSEIF ( repl(1:5) .EQ. "USER_" ) THEN
              IF ( repl .EQ. "USER_SCALE" ) THEN
	         CALL GET_USER_SCALE (cx, result)
	      ELSEIF ( repl(1:8) .EQ. "USER_OFF" ) THEN
                 CALL GET_USER_OFFSET (cx, result)
	      ELSE 
	         GOTO 5520
	      ENDIF
	      repl = TM_FMT( result, use_digits, 22, str_len )

* ... varname ,RETURN=dtype
	   ELSEIF ( repl .EQ. "DTYPE" .AND. 
     .              INDEX(cmnd(istart:iend),'.') .EQ. 0) THEN
              dset = cx_data_set(cx)
              IF (dset .EQ. pdset_irrelevant .OR. 
     .            dset .EQ. unspecified_int4) dset = pdset_uvars  ! try user vars

              varid = 0
              vartype = 0

              CALL CD_GET_VAR_ID(dset, cmnd(istart:iend), varid, status)

              IF (status .EQ. ferr_ok)  
     .          CALL CD_GET_VAR_TYPE (dset, varid, cmnd(istart:iend), 
     .                              vartype, status )

              IF (vartype.EQ.0 ) THEN  

*  evaluate expr to get type and translate ferret type to netcdf

	         CALL EVAL_EXPR (  cx_last,
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
              IF (cx_data_set(cx) .EQ. pdset_irrelevant) dset = pdset_uvars  ! user vars
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
     .              CALL BREAK_VARATTNAME( cmnd(istart:iend), dset, varname, 
     .                              attname,  varid, do_err, status )

                 ELSE
                    CALL CD_GET_VAR_ID (dset, cmnd(istart:iend), varid,
     .                  status)
                    
                    IF (status .NE. ferr_ok .AND. 
     .                  INDEX(cmnd(istart:iend),'.') .GT. 0) THEN
                       CALL BREAK_VARATTNAME( cmnd(istart:iend), dset,
     .                         varname, attname, varid, do_err, status )
                    ELSE
                       i2 = CGRID_SIZE ( cx )
                       repl = LEFINT(i2, str_len) 
                       GOTO 500
                    ENDIF

                 ENDIF
                 
                 CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, 
     .               status)
                 IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid,
     .               attid, aname,  attype, attlen, attoutflag, status )

                 IF (status .EQ. ferr_ok) THEN
                    IF (attlen .GT. 0) THEN
                       result = attlen
                       repl = TM_FMT( result, use_digits, 22, str_len )
                    ENDIF
	         ELSE 
	            GOTO 5520
                 ENDIF
              ENDIF

* ... RETURN=XMOD
	   ELSEIF ( repl .EQ. "XMOD" ) THEN
	      repl = ' '	! default
	      axis = CGRID_AXIS ( X_dim, cx )
	      IF ( axis.NE.mnormal) THEN
	         IF (line_modulo(axis)) THEN
                    result = line_modulo_len(axis)

! If axis has modulo=" ", and units are not degrees, mod len = axis len.
                    IF (result .EQ. 0) result = TM_WW_AXLEN(axis)
                    repl = TM_FMT( result, use_digits, 22, str_len)
                 ENDIF
	      ENDIF

* ... RETURN=TMOD
	   ELSEIF ( repl .EQ. "TMOD" ) THEN
	      repl = ' '	! default
	      axis = CGRID_AXIS ( T_dim, cx )
	      IF ( axis.NE.mnormal) THEN
	         IF (line_modulo(axis)) THEN
                    result = line_modulo_len(axis)

! If axis has modulo=" ", mod len = axis len.
                    IF (result .EQ. 0) result = TM_WW_AXLEN(axis)
                    repl = TM_FMT( result, use_digits, 22, str_len)
                 ENDIF
	      ENDIF

* ... RETURN=ISDEPTH
	   ELSEIF ( repl(1:3) .EQ. "ISD" ) THEN
	      repl = '0'	! default, pos=up, or not-depth.
	      axis = CGRID_AXIS ( Z_dim, cx )
	      IF ( axis.NE.mnormal) THEN
	         IF (line_direction(axis) .EQ. "UD") repl = '1'
	      ELSE 
	         repl = 'NORMAL'
	      ENDIF

* ... RETURN=ISLOADED
!	   ELSEIF ( repl(1:4) .EQ. "ISLO" ) THEN
!* see is an mr variable matching this cx is already in memory
!	      CALL FIND_MEM_VAR( cx, mr, search_by_value )
!	      IF ( mr.EQ.mr_nonexist ) THEN
!	         repl = '0'
!	      ELSE
!	         repl = '1'
!	      ENDIF
!	   
* ... RETURN=*SIZE, *START, *END, *UNITS, *AXIS, or *COORD for dsg datasets
	   ELSE	
* ... axis specific requests - must start with i,j,k,l,m,n or x,y,z,t,e,f
	      DO 220 idim = 1, nferdims
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
* Get axis name. If mode upcase is canceled, use original spelling if defined
		    orig_name = .FALSE.
		    CALL CHOOSE_LINE_NAME(axis, orig_name, repl)
	         ENDIF
	         need_formatting = .FALSE.

	      ELSEIF ( repl(2:5) .EQ. "COOR") THEN
	         grid = cx_grid(cx)
		 IF (TM_ITSA_DSG(grid)) THEN

		    dset = TM_DSG_DSET_FROM_GRID( grid )

		    IF (idim.LE.t_dim) THEN
		       repl = VAR_CODE( cat_file_var, dsg_coord_var(idim,dset))

		    ELSE
		       axis = CGRID_AXIS ( idim, cx )
		       IF (axis.EQ.mnormal .OR. axis.EQ.munknown) THEN
	                 repl = 'NORMAL'
		       ELSE
* Get axis name. If mode upcase is canceled, use original spelling if defined
		          orig_name = .FALSE.
		          CALL CHOOSE_LINE_NAME(axis, orig_name, repl)
		       ENDIF
		       need_formatting = .FALSE.
		    ENDIF

		 ELSE ! not dsg
		    GOTO 5530
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

* Get the units attribute from the dataset. 
* User-defined true-month axes are shown in days.
	         IF ( repl(1:8) .EQ. "TRUMONTH") THEN
	            dset = cx_data_set( cx )
		    aname = 'days'
c	            IF (dset .GT. pdset_irrelevant) THEN
c	               CALL GET_ORIGINAL_UNITS (dset, cx_grid(cx), aname, ucode)
c	               IF (ucode .NE. 0) repl = aname
c		    ENDIF
		 ENDIF

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
*       ndec is precision when positive, number of sig digits when negative
	              CALL TRANSLATE_TO_WORLD( ww, idim, cx_grid(cx),
     .				               -use_digits, repl )
!	            ENDIF
	         ENDIF
	      ENDIF
	   ENDIF
	   str_len = TM_LENSTR1(repl)

	ELSE
* single number answer desired
* ... evaluate the expression

	   cx_only = .FALSE.
	   CALL EVAL_EXPR (  cx_last, cmnd(istart:iend), 
     .        .TRUE., status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   cx = is_cx(1)	! always comes back as the first context
	   mr = is_mr(1)

* ... make sure the result is a single, valid value
	   
	   is_dsg = .FALSE.
	   IF ( CGRID_SIZE(cx) .NE. 1 ) THEN

*    For DSG grids the requested region may be just in the WW context in E
	      is_dsg = TM_ITSA_DSG( cx_grid(cx) )
	      IF (CGRID_SIZE(cx).GT.1 .AND. is_dsg ) THEN  
	         siz = 1
		 ifeature = 0
	         DO idim = x_dim, t_dim
	      	      siz = siz * (
     .                 cx_hi_ss(cx,idim)  - cx_lo_ss(cx,idim) + 1 )
	      	      IF (cx_lo_ss(cx, idim) .NE. unspecified_int4) 
     .                 ifeature = cx_lo_ss(cx,idim)
	         ENDDO
	         siz = siz * ( 
     .               cx_hi_ww(e_dim,cx) - cx_lo_ww(e_dim,cx) + 1 )
	         siz = siz * ( 
     .               cx_hi_ss(cx,f_dim) - cx_lo_ss(cx,f_dim) + 1 )
	         IF (siz .NE. 1) GOTO 5200

		 IF (ifeature.EQ.0 .AND. 
     .               cx_hi_ww(e_dim,cx).NE.unspecified_val8) 
     .                 ifeature = cx_hi_ww(e_dim,cx)
		 IF (ifeature.EQ.0 .AND. 
     .               cx_hi_ww(f_dim,cx).NE.unspecified_val8) 
     .                 ifeature = cx_hi_ww(f_dim,cx)
		 IF (ifeature .EQ. 0) ifeature = 1
	      ELSE
	         IF ( CGRID_SIZE(cx) .NE. 1 ) GOTO 5200	! result not a scalar
	      ENDIF

	   ENDIF

* ... extract the result scalar from the (possibly larger) memory variable
	   CALL CREATE_TEMP_MEM_VAR( cx, mr_temp, status )
	   IF ( status .NE. ferr_ok ) RETURN

	   CALL COPY_GRID( memry(mr)%ptr, mr,
     .                     memry(mr_temp)%ptr, mr_temp )

           IF (is_dsg) CALL DSG_ONE_INSTANCE_VAL (mr, mr_type(mr_temp), 
     .                           ifeature, result, repl, max_len, str_len)

	   IF (mr_type(mr_temp) .EQ. ptype_string ) THEN

* ... get one string


	     IF (.NOT. is_dsg) repl =  GET_STRING_ELEMENT(
     .				mr_lo_s1(mr_temp), mr_hi_s1(mr_temp),
     .				mr_lo_s2(mr_temp), mr_hi_s2(mr_temp),
     .				mr_lo_s3(mr_temp), mr_hi_s3(mr_temp),
     .				mr_lo_s4(mr_temp), mr_hi_s4(mr_temp),
     .				mr_lo_s5(mr_temp), mr_hi_s5(mr_temp),
     .				mr_lo_s6(mr_temp), mr_hi_s6(mr_temp),
     .				memry(mr_temp)%ptr,
     .				cx_lo_s1(cx), cx_lo_s2(cx),
     .				cx_lo_s3(cx), cx_lo_s4(cx),
     .				cx_lo_s5(cx), cx_lo_s6(cx),
     .				max_len, str_len)
	   ELSE

	     IF (.NOT. is_dsg) result =  memry(mr_temp)%ptr(1)

	     IF ( result .EQ. mr_bad_data(mr) ) THEN
	        repl = bad_str
	        str_len = TM_LENSTR1(bad_str)
	     ELSE
                repl = TM_FMT(result, use_digits, 48, str_len)
* Did the large allowed string length result in a string like 0.00000001 or 7.499999999999 ?  
* If so see if we can return a nicer string that is essentially the same numeric value.
* First try 24, then 16.
                shorterlen = 24
  400           dotidx = INDEX(repl, '.')
                IF ( (dotidx .GT. 0) .AND. (use_digits .GT. 0) .AND.
     .               ( (INDEX(repl,'000000') .GT. dotidx) .OR.
     .                 (INDEX(repl,'999999') .GT. dotidx) ) ) THEN
                   buff = TM_FMT(result, use_digits, shorterlen, itmp)
                   IF ( buff(1:1) .NE. '*' ) THEN
                      READ (buff,*) aval
                      IF ( TM_DFPEQ(result, aval) ) THEN
                         repl = buff
                         str_len = itmp
                         IF ( shorterlen .GT. 16 ) THEN
                            shorterlen = 16
                            GOTO 400
                         ENDIF
                      ENDIF
                   ENDIF
                ENDIF
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

* restore the state of uvar grids etc changed here for returns 
* that need the grid, such as *size, shape. The uvar.GT.0 is 
* checking for "packed" uvar-context info as set in const_var_context

        IF (cx_only) THEN
	   CALL RESET_UVAR_GRIDS
	   CALL FREE_LINE_DYNMEM(uvar_cx_only_lm)

	   cx_only = .FALSE.

	ENDIF

	RETURN

* error exits
 5050	CALL RELEASE_DYN_WORK_SPACE
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
     .       'maximum precision is 16 digits: '
     .					//risc_buff, *5000)
 5520   risc_buff = cmnd(i:qual_end)
	CALL ERRMSG( ferr_invalid_command, status,
     .     return_errmsg//pCR//risc_buff, *5000)

 5530   risc_buff = cmnd(i:qual_end)
	CALL ERRMSG( ferr_invalid_command, status,
     .       'Variable is not on a DSG grid in a DSG dataset: '
     .					//risc_buff, *5000)
	END
