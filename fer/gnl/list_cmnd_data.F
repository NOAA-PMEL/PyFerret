        SUBROUTINE LIST_CMND_DATA( grid_data, mr, cx,
     .                             ax_buff, heading, head_enh, recsofar,
     .                             perm, clobber, dods_file, status )

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
* list a gridded variable stored in memory - 0 to 6 dimensions
* Note: the looping/axis labelling procedures of LIST_MULTI are cleaner and
*	more efficient than this - this could be improved easily (*sh* 7/90)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/16/86
* V200:  6/22/89 - major changes to accomodate 4D memory resident variables
*		 - handle data sets that are indicated "irrelevant"
*	  9/1/89 - accept permutation specification
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN
*	12/19/89 - eliminated the possibility of multiple data set variables
*	  7/9/90 - extracted code fragments to subroutines MINMAX, VAR_DATA_FMT
*		   and ROW_COORD_FMT to be shared with LIST_MULTI
*		   changed POS_LAB call
* V301: 11/1/93 - change to WRITE(ERR=...) processing for DEC compilers 
* V313: 31/10/94 - size of row_ss_fmt increased due to ROW_COORD_FMT chg
* V320: 10/5/94 - added code for /HEADING=enhanced
*	11/9/94 - replaced all WRITE statements with SPLIT_LIST 
*	12/28/94- added LIST/FORMAT=STREAM (add'l argument "recsofar")
*		  and added units and T0 to /HEADING=enhanced (/xtm_grid.cmn)
*	3/2/95  - fixed bug in SIZE output for /HEADING=enhanced
* V420  9/28/95 - added tab and comma-delimited output format
*		- line_buff increased hugely (from 132) to accomodate
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
* V500: *kob* 3/99 - up VAR_UNITS from 32 to 64 
* V530:  8/00 *sh* - added support for string variables
* V530:  3/01 *acm*- Put the calendar name in the heading if TIME not
*		     otherwise listed, and if non-gregorian calendar.
* V540: 11/01 *sh* - added LIST/WIDTH=
*	           - expanded and reorganized the LIST headng
* V552   2/03 *acm*- Increase line_buff to 10000
* v553   9/03 *kob*- Increase GET_STRING_ELEMENT to 512 
* V554   3/04 *acm*-For REPEAT/RANGE vars, add check for dset = unspecified_int4

* V554   9/03 *acm* - add output in /DODS binary format, see the flag do_dods 
*                     change this comment, so that the checkin under branch FDSv1
*                     will occur.
* v560   4/04 *acm* Very wide output needs a fix in the format specifier that is written.
* V580   9/04 *acm* call ndig_coords to compute the precision needed to list the
*                   coordinates in tab and comma delimited formats.  Note that if
*                   precision more than 0.001 is needed for longitude and latitude
*                   axes, then a change needs to be made in translate_to_world.
* V581*acm*  6/05 - For fix to bug 1271, add flag line_shift_origin
*            and restore original t0 on output
* V600: *acm* 8/05 - Add new qualifier LIST/NOROWHEAD to remove labels that
*                    start each row labelling the coordinate values. (bug 1273)
* V650: *acm* 2/10 - Clean up line with coordinates on listings with tabs/commas
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* V691+ *acm* 9/14 Ticket 2199: for string variables don't list a BAD FLAG in the header.
* V71   *acm* 11/16 For true-month axes, call TM_WORLD to write any true-month
*                  time coordinates as days.
* V71   *acm* 01/17 Ticket 2492. For single-variable LIST/FORMAT=comma or /FORMAT=tab listings, 
*                   /NOROW had no effect; and there is an extra longitude location label below 
*                   the header even when the longitude has been averaged away.
* v74  *acm* 3/18  Issue 1856, changes to allow for timesteps in fractional seconds
* V740 *acm*  3/18 Issue 950: Allow simplified I formats
* V741 *acm*  4/18 Issue 1870: Check Format more closely when finding an I Format
* V741 *acm*  4/18 Issue 1878: Ferret error if the WRITE results in an error
* V74+ *acm* 10/18 routine minmax returns count of valid data
* V751 6/19 *acm* TM_FMT rewrite: formatting of lon/lat coordinate strings
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	include 'xtm_grid.cmn_text'		! for line_units and t0
	external xdset_info_data
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'slash.parm'
	include 'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xprog_state.cmn'
	include 'xcontext.cmn'
	include 'xtext_info.cmn'
	include 'xinterrupt.cmn'
	include 'xrisc.cmn'
	include 'calendar.decl'
	include 'calendar.cmn'

* local parameter declarations:
	INTEGER		 min_field_width, max_line_len
	PARAMETER     ( min_field_width = 4,
     .			max_line_len = 10000 )

* calling argument declarations:
	LOGICAL		heading, head_enh, clobber
	INTEGER		mr, cx, perm(nferdims), recsofar, status
	REAL grid_data(m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit,m1loe:m1hie,m1lof:m1hif),
     .       ax_buff(*)
        CHARACTER*(*) dods_file
	
* local variable declarations:
	LOGICAL		GEOG_LABEL, TM_DFPEQ, ITS_FMRC, TM_DIGIT,
     .			line_too_long, del_col_changed, all_bad,
     .			valid(nferdims), need_doc(nferdims), span(nferdims), do_unform,
     .			do_stream, do_comma_del, do_tab_del, itsa_string,
     .			fortran_selfdoc, do_auto_fmt, do_dods, norow, has_int
	INTEGER		TM_LENSTR, TM_LENSTR1, CX_DIM_LEN,
     .			FIELD_WIDTH, MGRID_SIZE, TM_LOC_STRING, 
     .			GET_MAX_C_STRING_LEN, TM_GET_CALENDAR_ID, STR_SAME,
     .			width, w0, grid, dlen, prec_digits, ndigits,
     .			ndim, lo(nferdims), hi(nferdims), del(nferdims),
     .			i, j, k, l, m, n, npts, line_length, dset,
     .			temp, num_fld_est, nleft, nright, alen, num_fields,
     .			idim, vlen, istart, iend, ifield,
     .			cleft, usable_line_len, max_fields, row_ax,
     .			col_ax, places, col_dec, row_dec, row_wld, 
     .			numbers, first, slen, llen, axis, lp, cal_id,
     .                  wkblk, ndig(nferdims), ival, num_it, slen1, 
     .                  i1, nok, ndx, n_alloc
	CHARACTER*255	VAR_TITLE, VAR_TITLE_MOD, PAREN_SURROUND, vname_buff
* V500: *kob* 3/99 - up VAR_UNITS from 32 to 64 
	CHARACTER	AX_TITLE*32, VAR_UNITS*64, LEFINT*8, TM_FMT*48,
     .			GET_STRING_ELEMENT*512, fmtline*16,
     .			line_buff*10000, buff*128, b1*1, out_type*3,
     .			good_data_form*10, row_ss_fmt*14, tab_or_comma*1,
     .			bad_data_form*10, col_head_format*32, cal_name*32
	REAL		BOX_SIZE, big, small, biggest, bad_data, value
	REAL*8		TM_WORLD, val8, val_last, diff_min, diff_ave, count


	REAL*4, DIMENSION(:), ALLOCATABLE :: floatdata
	INTEGER(kind = 4), DIMENSION(:), ALLOCATABLE :: intdata
	INTEGER(kind = 2), DIMENSION(:), ALLOCATABLE  :: shortdata
	INTEGER(kind = 1), DIMENSION(:), ALLOCATABLE  :: bytedata

* convenience equivalence:* convenience equivalence:
        INTEGER         lo1, lo2, lo3, lo4, lo5, lo6, 
     .                  hi1, hi2, hi3, hi4, hi5, hi6, 
     .                  del1, del2, del3, del4, del5, del6
        EQUIVALENCE     (lo1, lo(1)), (lo2, lo(2)), (lo3, lo(3)),  
     .                  (lo4, lo(4)), (lo5, lo(5)), (lo6, lo(6)),
     .                  (hi1, hi(1)), (hi2, hi(2)), (hi3, hi(3)),  
     .                  (hi4, hi(4)), (hi5, hi(5)), (hi6, hi(6)),
     .                  (del1,del(1)),(del2,del(2)),(del3,del(3)),
     .                  (del4,del(4)),(del5,del(5)),(del6,del(6))


* initialize
	status		= ferr_ok
	grid		= mr_grid( mr )
	bad_data	= mr_bad_data( mr )
	line_too_long	= .FALSE.
	del_col_changed	= .FALSE.
	width		= min_field_width
        do_unform       = list_fmt_type .EQ. plist_unformatted
        do_dods         = list_fmt_type .EQ. plist_dods
        do_stream       = list_fmt_type .EQ. plist_stream
        do_comma_del    = list_fmt_type .EQ. plist_comma_del
        do_tab_del      = list_fmt_type .EQ. plist_tab_del
        do_auto_fmt     = list_fmt_type .EQ. plist_default
	fortran_selfdoc = head_enh .AND. list_format_given
	itsa_string     = mr_type(mr) .EQ. ptype_string
        norow = qual_given(slash_list_norow) .GT. 0
	good_data_form = ''
	good_data_form = ''
	
	n_alloc = -1

* determine precision for outputting values
	lp = qual_given( slash_list_precision ) 
	IF ( lp .GT. 0 ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(lp):qual_end(lp)),
     .                     value, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
*          negative values for precision permitted (number of digits after the decimal point)
           prec_digits = NINT(value)
           IF ( prec_digits .LT. -16 ) THEN
              prec_digits = -16
           ELSE IF ( prec_digits .GT. 16 ) THEN
              prec_digits = 16
           ENDIF
	ELSE
	   prec_digits = list_digits
	ENDIF

* determine maximum width of the output records
	lp = qual_given( slash_list_width )
	IF ( lp .GT. 0 ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(lp):qual_end(lp)),
     .                     value, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
	   i = value
	   line_length  = MIN(ABS(i),max_line_len)
	ELSE
	   line_length  = 130
	ENDIF

* determine row and column axes and how many axes are "significant"
	row_ax = perm(1)
	col_ax = perm(2)
	ndim = 0
	DO 10 ndim = nferdims, 1, -1
	   IF ( CX_DIM_LEN( perm(ndim), cx ) .GT. 1 ) GOTO 20
 10	CONTINUE
	ndim = 0
 20	DO 30 idim = 1, nferdims
	   span(idim) = CX_DIM_LEN( perm(idim), cx ) .GT. 1
	   valid(idim) = cx_lo_ss( cx, perm(idim) ) .NE. unspecified_int4
 30	CONTINUE


* * * * * * WRITE INTRODUCTORY LINES * * * * *
	IF ( do_unform .OR. do_stream .OR. do_dods .OR.
     .       .NOT.heading ) GOTO 200
* ... first the variable name with units
	vname_buff   = VAR_TITLE( cx )
	vlen   = TM_LENSTR1( vname_buff )
        IF (INDEX(VAR_UNITS(cx), 'truemonth') .GT. 0 .OR. 
     .      INDEX(VAR_UNITS(cx), 'TRUEMONTH') .GT. 0 ) THEN
           vname_buff   = vname_buff(:vlen)//' '//
     .				PAREN_SURROUND('day', alen)
        ELSE
	   vname_buff   = vname_buff(:vlen)//' '//
     .				PAREN_SURROUND(VAR_UNITS(cx), alen)
        ENDIF
        
	CALL SPLIT_LIST(pttmode_explct, list_lun,
     .				'             VARIABLE : '//vname_buff, 0)

* ... qualify title with modification line ?
	vname_buff = VAR_TITLE_MOD( cx )
	vlen   = TM_LENSTR( vname_buff )
	IF ( vlen .GT. 0 ) CALL SPLIT_LIST( pttmode_explct, list_lun,
     .			'                        '//vname_buff(1:vlen),
     .				      vlen+24)

* ... then DATA SET
	dset = cx_data_set(cx)
	IF ( dset .NE. pdset_irrelevant .AND.        ! for REPEAT/RANGE vars, add 
     .       dset .NE. unspecified_int4) THEN        ! check for unspecified_int4
* ... title
	   CALL GET_DSET_TITLE( cx_data_set(cx), line_buff, slen )
	   IF (line_buff .NE. ' ') THEN
	     CALL SPLIT_LIST(pttmode_explct, list_lun,
     .			'             DATA SET : '//
     .			 line_buff(1:slen), slen+24)
	   ENDIF
* ... find split point between filename and path
	   slen = TM_LENSTR1(ds_des_name(dset))
	   DO i = slen, 1, -1
	      IF ( ds_des_name(dset)(i:i) .EQ. '/' ) GOTO 34
	   ENDDO
	   i = 0
* ... filename
 34	   CALL SPLIT_LIST(pttmode_explct, list_lun,
     .			'             FILENAME : '//
     .			ds_des_name(dset)(i+1:slen), 24+(slen-i))
* ... DODS URL BASE or ABSOLUTE PATH

	   IF (i.GT.0 .AND.  ds_des_name(dset)(1:1).NE.'.')
     .			CALL SPLIT_LIST(pttmode_explct, list_lun,
     .			'             FILEPATH : '//
     .			 ds_des_name(dset)(:i), i+24)
	ENDIF

* ... BAD FLAG: xxx
	IF (.NOT.fortran_selfdoc) THEN
	   IF (.NOT.do_auto_fmt) THEN
	     line_buff = ' '
	     WRITE ( line_buff, '(1PG14.7)' ) cx_bad_data(cx)
	     line_buff = TM_FMT(cx_bad_data(cx),prec_digits,16,vlen)
	     IF (itsa_string) line_buff = ' '
	     CALL SPLIT_LIST( pttmode_explct, list_lun,
     .			'             BAD FLAG : '//
     .		        line_buff, 38 )
	   ENDIF
* ... SUBSET:   e.g. "Nx by Ny (longitude-latitude)"
	   line_buff = '             SUBSET   : '
	   l = 24
	   DO idim = 1, nferdims
	      IF ( span(idim) ) THEN
	         i = perm(idim)
	         j = CX_DIM_LEN ( i, cx )
	         line_buff = line_buff(:l)//LEFINT( j, vlen )
	         l = l + vlen
	         line_buff = line_buff(:l)//' by '
	         l = l + 4
	      ENDIF
	   ENDDO
	   IF ( l.GT.24) THEN
	      line_buff(l-3:l+7) = ' points (  '
	      l = l + 5
	      DO idim = 1, nferdims
	         IF ( span(idim) ) THEN
	            line_buff = line_buff(:l)//
     .				AX_TITLE(perm(idim), cx_grid(cx), vlen)
	            l = l + vlen
	            line_buff = line_buff(:l)//'-'
	            l = l + 1
	         ENDIF
	      ENDDO
	      line_buff(l:l) = ')'
	      CALL SPLIT_LIST(pttmode_explct, list_lun,
     .			      line_buff(:l), l)
	   ENDIF
	ENDIF

* ... which axes require labels
	DO idim = 1, nferdims
	   need_doc(idim) = .TRUE.
 	ENDDO
	IF ( .NOT.(do_unform.OR.do_dods.OR.list_format_given) ) THEN
	   DO idim = 1, ndim
	      need_doc(idim) = CX_DIM_LEN( perm(idim), cx ) .EQ. 1
 	   ENDDO
	ENDIF

* ... if not indicated in time label show a nonstandard calendar name
	cal_name = line_cal_name( grid_line(t_dim, grid) )
	cal_id = TM_GET_CALENDAR_ID (cal_name)
	DO i = 1, nferdims
	   idim = perm(i)
	   IF ( .NOT.need_doc(i) .AND. idim.EQ.t_dim .AND.
     .	     cal_id .GT. gregorian .AND. cal_id .LE. max_calendars) THEN
	      slen = TM_LENSTR(cal_name)
	      CALL SPLIT_LIST( pttmode_explct, list_lun,
     .                  '             CALENDAR : '
     .			//cal_name(:slen), slen+24)
	   ENDIF
	ENDDO

* ... region information
	DO 60 i = 1, nferdims
	   idim = perm(i)
	   IF ( need_doc(i) ) THEN
	      IF ( cx_lo_ww( idim,cx ) .NE. unspecified_val8 ) THEN 
	         CALL ASCII_LIMITS( cx, idim, line_buff, vlen )
	         col_head_format = AX_TITLE( idim, grid, alen )
	         IF (alen .LT. 9) alen = 9
	         CALL SPLIT_LIST( pttmode_explct, list_lun,
     .			'             '//col_head_format( :alen )//': '
     .			//line_buff(1:vlen), alen+vlen+15)
	      ENDIF
	   ENDIF
 60	CONTINUE

* ... T-F plane coaching
*    If this variable is FMRC, then add annotations to disambiguate the 2 time axes
*      "Initialization time increases across row. Verification time increases down page."
        IF ( ITS_FMRC(grid)
     . .AND. (row_ax.EQ.t_dim .OR. row_ax .EQ. f_dim
     .   .OR. col_ax.EQ.t_dim .OR. col_ax .EQ. f_dim) ) THEN
           line_buff = ' '
           alen = 0
           IF (row_ax.EQ.t_dim .OR. row_ax .EQ. f_dim) THEN
              IF (row_ax.EQ.t_dim) THEN
                 line_buff = 'Verification'
                 alen = 14
              ELSE
                 line_buff = 'Initialization'
                 alen = 16
              ENDIF
              line_buff(alen:) = "time increases across row."
              alen = alen + 26
           ENDIF
           IF (col_ax.EQ.t_dim .OR. col_ax .EQ. f_dim) THEN
              IF (alen .GT. 0) alen = alen + 1
              IF (col_ax.EQ.t_dim) THEN
                 line_buff(alen+1:) = 'Verification'
                 alen = alen + 14
              ELSE
                 line_buff(alen+1:) = 'Initialization'
                 alen = alen + 16
              ENDIF
              line_buff(alen:) = "time increases down page."
              alen = alen + 25
           ENDIF
           CALL SPLIT_LIST(pttmode_explct,list_lun,
     .      '             '//line_buff,alen+13)
        ENDIF

* * * * * DETERMINE LIST LIMITS * * *
 200	DO 210 i = 1, nferdims
	   idim = perm(i)
	   lo(i) = cx_lo_ss( cx, idim )
	   hi(i) = cx_hi_ss( cx, idim )
	   del(i)= 1
 210	CONTINUE



* * * * *  ENHANCED HEADING STUFF (10/94) * * * * *
	IF ( fortran_selfdoc ) THEN
* ... GEOMETRY: xxx
	   line_buff = ' '
	   i = 1	   
	   DO 220 idim = 1, nferdims
	      IF ( span(idim) ) THEN
	         line_buff(i:i) = ww_dim_name(perm(idim))
	         i = i + 1
	      ENDIF
 220	   CONTINUE
	   CALL SPLIT_LIST(pttmode_explct, list_lun,
     .				'             GEOMETRY: '//
     .			 	line_buff(:i), i+23)

* ... SIZE: xxx
	   line_buff = ' '
	   l = 1
	   DO 230 idim = 1, nferdims
	      IF ( span(idim) ) THEN
	         i = perm(idim)
	         j = CX_DIM_LEN ( i, cx )
	         line_buff = line_buff(:l)//LEFINT( j, vlen )
	         l = l + vlen + 2
	      ENDIF
 230	   CONTINUE
	   CALL SPLIT_LIST(pttmode_explct, list_lun,
     .				'             SIZE: '//
     .			 	line_buff(:l), l+19)

* ... FORTRAN FORMAT: xxx
	   IF ( list_format .EQ. '(5(1PG12.5))' ) THEN
	      CALL SPLIT_LIST(pttmode_explct, list_lun,
     .		'             FORTRAN FORMAT: (5F12.5)', 37 )
	   ELSE
	      CALL SPLIT_LIST(pttmode_explct, list_lun,
     .		'             FORTRAN FORMAT: '//list_format, 0)
	   ENDIF


* ... MISSING VALUE FLAG: xxx
	   line_buff = ' '
	   WRITE ( line_buff, 3062 ) cx_bad_data(cx)
 3062	   FORMAT( 13x,'MISSING VALUES FLAG: ',1PG14.7)
	   CALL SPLIT_LIST( pttmode_explct, list_lun, line_buff, 0 )

* ... axis COORDINATES:
	   DO 250 idim = 1, nferdims
	      IF ( span(idim) ) THEN
	         line_buff = ' '
	         i = perm(idim)
	         axis = grid_line(i, cx_grid(cx) )
	         WRITE ( line_buff, 3065 ) ww_dim_name(i)
 3065	         FORMAT(A1,' COORDINATES: ')
	         line_buff(16:) = line_units(axis)
	         alen = TM_LENSTR1( line_buff )
	  	 IF (  i .EQ. t_dim ) THEN		! add T0=
	            IF ( GEOG_LABEL(i,cx_grid(cx)) ) THEN 
	               line_buff(alen+2:) =  '(T0='//line_t0(axis)
                       IF (line_shift_origin(axis)) 
     .                  line_buff(alen+2:) =  '(T0=01-JAN-0001 00:00:00'
	               alen = TM_LENSTR1( line_buff )
	               line_buff(alen+1:) =  ')'
	               alen = alen + 1
	            ENDIF
	         ENDIF
	         CALL SPLIT_LIST(pttmode_explct,list_lun,line_buff,alen)
	         npts =  hi(idim)-lo(idim)+1
                 CALL MAKE_AXIS(cx_grid(cx),i,lo(idim),hi(idim),ax_buff)
! Note: FORMATTED output cannot be passed through SPLIT_AXIS because a
!	single WRITE may imply many records of output
	         WRITE ( list_lun, list_format ) (ax_buff(j),j=1,npts)
	      ENDIF
 250	   CONTINUE 


* ... DATA:
	     CALL SPLIT_LIST(pttmode_explct, list_lun, 'DATA:', 5 )
	ENDIF

* * * * * LIST DATA UNFORMATTED OR WITH USER FORMAT (no frills)


	IF ( do_dods) THEN
           CALL WRITE_DODS_FMT (grid_data, 
     .           m1lox,m1hix, m1loy,m1hiy, m1loz,m1hiz, m1lot,m1hit, 
     .           ax_buff, dods_file, clobber, status)

* done - 
	   RETURN
        ENDIF

	IF ( do_unform
     .	.OR. do_stream
     .	.OR. list_format_given ) THEN

* Limited handling of integer formats -- write single number per record
	   has_int = (TM_LOC_STRING (list_format, 'I', 1) .GT. 0)
	   IF (has_int) THEN

* Make sure the I is a format spec, not within a string in the format  
	      slen1 = TM_LENSTR1(list_format)
	      CALL PARSE_COMMA_LIST (list_format, 2, slen1-1, max_item_list,
     .	          num_it, item_start, item_end, status)
	      has_int = .FALSE.
	      DO i = 1, num_it
	         buff = list_format(item_start(i):item_end(i))
		 
		 ! Is this item from the fmt statement a quoted string?
		 i1 = item_start(i) - 1
		 IF (i1.GE.1) THEN         
		    b1 = list_format(i1:i1)
		    IF ( b1 .EQ. '"' .OR.  b1 .EQ. "'") CYCLE
		 ENDIF

* Not a string, is the first letter an I? 
		 has_int = (STR_SAME(buff(1:1), 'i') .EQ. 0)
	      ENDDO
	   ENDIF

	   IF (has_int) THEN
	      ival = TM_LOC_STRING (list_format, 'I', 1) - 1
	      IF (TM_DIGIT( list_format(ival:ival)) ) GOTO 5020
	   ENDIF

* Have already checked  in XEQ_LIST for allowed type specified.
	   out_type = 'DOU'
	   IF ( do_stream) THEN
	      i = qual_given( slash_list_outtype )

	      IF ( i .GT. 0 ) THEN
                 IF (do_unform .OR. do_stream) THEN
	            CALL EQUAL_STRING(
     .                  cmnd_buff(qual_start(i):qual_end(i)),
     .                  out_type, status )
	            IF ( status .NE. ferr_ok ) GOTO 5000
                 ENDIF
	      ENDIF
	   ENDIF

	   IF (out_type(1:3) .EQ. 'FLO') THEN
	      n_alloc = hi1 - lo1 + 1 
	      ALLOCATE(floatdata(n_alloc))
	   ENDIF
	   IF (out_type(1:3) .EQ. 'INT') THEN
	      n_alloc = hi1 - lo1 + 1 
	      ALLOCATE(intdata(n_alloc))
	   ENDIF
	   IF (out_type(1:3) .EQ. 'SHO') THEN
	      n_alloc = hi1 - lo1 + 1 
	      ALLOCATE(shortdata(n_alloc))
	   ENDIF
	   IF (out_type(1:3) .EQ. 'BYT') THEN
	      n_alloc = hi1 - lo1 + 1 
	      ALLOCATE(bytedata(n_alloc))
	   ENDIF

	   DO n = lo6, hi6, del6
	   DO m = lo5, hi5, del5
	   DO l = lo4, hi4, del4
	   DO k = lo3, hi3, del3
	   DO j = lo2, hi2, del2
	      IF ( interrupted ) RETURN
! ----- start string handler -------------
	      IF (itsa_string) THEN
	         llen = 0
	         DO 270 i = lo1, hi1, del1
	            line_buff(llen+1:) = GET_STRING_ELEMENT(
     .			m1lox,m1hix,m1loy,m1hiy,m1loz,m1hiz,m1lot,m1hit,
     .			m1loe,m1hie,m1lof,m1hif,grid_data,i,j,k,l,m,n,
     .			max_line_len,slen )
	            llen = MIN(max_line_len-1, llen+slen)
	            IF (do_stream) THEN
	               llen = llen + 1
	               line_buff(llen:llen) = CHAR(0) ! NULL terminator
	            ENDIF
 270		 CONTINUE
	         IF ( do_unform ) THEN
	            WRITE (list_lun, ERR=5010) line_buff(:llen)
	         ELSEIF ( do_stream ) THEN
	            recsofar = recsofar + 1
	            WRITE (list_lun, REC=recsofar, ERR=5010)
     .						 line_buff(:llen)
	         ELSE
! Note: FORMATTED output cannot be passed through SPLIT_AXIS because a
!	single WRITE may imply many records of output
	            DO 272 i = lo1, hi1, del1
	               line_buff = GET_STRING_ELEMENT(
     .			m1lox,m1hix,m1loy,m1hiy,m1loz,m1hiz,m1lot,m1hit,
     .			m1loe,m1hie,m1lof,m1hif,grid_data,i,j,k,l,m,n,
     .			max_line_len,slen )
	               WRITE (list_lun,list_format, ERR=5010)
     .							line_buff(:slen)
 272		    CONTINUE
	         ENDIF
	         CYCLE
	      ENDIF  ! string
! ----- end string handler -------------
	      IF ( do_unform ) THEN
	         WRITE (list_lun, ERR=5010)
     .			( grid_data(i,j,k,l,m,n),i = lo1, hi1, del1 )

	      ELSEIF ( do_stream ) THEN
	         
	         IF (out_type(1:3) .EQ. 'FLO') THEN
		    ndx = 1
		    DO i = lo1, hi1, del1
		       floatdata(ndx) = grid_data(i,j,k,l,m,n)
		       ndx = ndx + 1
		    ENDDO

	            DO i = 1, n_alloc, del1
	               recsofar = recsofar + 1
	               WRITE (list_lun, REC=recsofar, ERR=5010)
     .						floatdata(i)
	            ENDDO
	         ELSEIF (out_type(1:3) .EQ. 'INT') THEN
		    ndx = 1
		    DO i = lo1, hi1, del1
		       intdata(ndx) = grid_data(i,j,k,l,m,n)
		       ndx = ndx + 1
		    ENDDO

	            DO i = 1, n_alloc, del1
	               recsofar = recsofar + 1
	               WRITE (list_lun, REC=recsofar, ERR=5010)
     .						intdata(i)
	            ENDDO
		    
	         ELSEIF (out_type(1:3) .EQ. 'SHO') THEN
		    ndx = 1
		    DO i = lo1, hi1, del1
		       shortdata(ndx) = grid_data(i,j,k,l,m,n)
		       ndx = ndx + 1
		    ENDDO

	            DO i = 1, n_alloc, del1
	               recsofar = recsofar + 1
	               WRITE (list_lun, REC=recsofar, ERR=5010)
     .						shortdata(i)
	            ENDDO
	         ELSEIF (out_type(1:3) .EQ. 'BYT') THEN
		    ndx = 1
		    DO i = lo1, hi1, del1
		       bytedata(ndx) = grid_data(i,j,k,l,m,n)
		       ndx = ndx + 1
		    ENDDO

	            DO i = 1, n_alloc, del1
	               recsofar = recsofar + 1
	               WRITE (list_lun, REC=recsofar, ERR=5010)
     .						bytedata(i)
	            ENDDO
	         ELSE
	            DO i = lo1, hi1, del1
	               recsofar = recsofar + 1
	               WRITE (list_lun, REC=recsofar, ERR=5010)
     .						grid_data(i,j,k,l,m,n)
	            ENDDO
		 ENDIF
	      ELSE
	         IF (has_int) THEN
		    DO i = lo1, hi1, del1
		       ival = INT(grid_data(i,j,k,l,m,n)) 
		       WRITE (list_lun, list_format) ival
		    ENDDO
		 ELSE
	            WRITE (list_lun,list_format, ERR=5010)
     .			( grid_data(i,j,k,l,m,n),i = lo1, hi1, del1 )
     		 ENDIF

	      ENDIF
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 	   ENDDO
 
	   IF (n_alloc .GE. 1) THEN
	       IF (out_type(1:3) .EQ. 'FLO') DEALLOCATE(floatdata)
	       IF (out_type(1:3) .EQ. 'INT') DEALLOCATE(intdata)
	       IF (out_type(1:3) .EQ. 'SHO') DEALLOCATE(shortdata)
	       IF (out_type(1:3) .EQ. 'BYT') DEALLOCATE(bytedata)
	   ENDIF

* done - go home
	   RETURN

* 9/95 - COMMA OR TAB DELIMITED OUTPUT ... some frills  but not too many
* (note that indent levels aren't changed for k and l loops)
	ELSEIF ( do_comma_del .OR. do_tab_del ) THEN
* ... column headings for tab and comma-delimited output (9/95)
	   IF ( do_comma_del ) THEN
	      tab_or_comma = ','
	   ELSEIF( do_tab_del ) THEN
	      tab_or_comma = CHAR(9)
	   ENDIF
	   IF ( do_comma_del) 
     .       line_buff(1:line_length) = '   '
	   IF ( do_tab_del) 
     .       line_buff(1:line_length) = '  '//tab_or_comma
	   alen = 3

           IF (perm(2) .EQ. 4) THEN ! time axis on the left      
	      IF ( do_comma_del) 
     .          line_buff(1:line_length) = '          '
	      IF ( do_tab_del) 
     .          line_buff(1:line_length) = '         '//tab_or_comma
	      alen = 10
           ENDIF

* Get number of decimal places for formatting the row coordinates

           ndig(row_ax)= ax_dec_pt(row_ax)

           CALL NDIG_COORDS (grid, row_ax, lo1, hi1, del1, 
     .        bad_data, ndig(row_ax), ax_dec_pt(row_ax) )

* Get number of decimal places for formatting the column coordinates

           ndig(col_ax) = ax_dec_pt(col_ax)
           CALL NDIG_COORDS (grid, col_ax, lo2, hi2, del2, 
     .        bad_data, ndig(col_ax), ax_dec_pt(col_ax) )


* Get number of decimal places for formatting coordinate labels

           ndig(perm(3)) = ax_dec_pt(perm(3))
	   IF ( ndim .GE. 3 .AND. valid(3) .AND. 
     .          ax_dec_pt(perm(3)) .GE. 0 ) 
     .        CALL NDIG_COORDS (grid, perm(3), lo3, hi3, del3, 
     .           bad_data, ndig(perm(3)), ax_dec_pt(perm(3)) )

           ndig(perm(4)) = ax_dec_pt(perm(4))
	   IF ( ndim .GE. 4 .AND. valid(4) ) 
     .        CALL NDIG_COORDS (grid, perm(4), lo4, hi4, del4, 
     .           bad_data, ndig(perm(4)), ax_dec_pt(perm(4)) )

* Skip this level of coordinate labels when they said /NOHEAD.
* Do not make spurious labels from an  index=-999.
	   IF (heading .AND. lo1.NE.unspecified_int4) THEN
	      DO 330 i = lo1, hi1, del1
	         val8 = TM_WORLD( i, grid, row_ax, box_middle )
	         width = FIELD_WIDTH(val8, grid, row_ax, ndig(row_ax),
     .					numbers)
	         alen = alen + width
	         IF ( alen .GT. max_line_len ) THEN
	            line_buff(max_line_len-18:max_line_len) = 
     .			'** line too long **'
	            alen = max_line_len
	            GOTO 331
	         ENDIF
	      
	         CALL TRANSLATE_TO_WORLD
     .		      ( val8,
     .			row_ax,
     .			grid,
     .			ndig(row_ax),
     .			line_buff(alen-(width-1):alen) )
	         IF ( i .LT. hi1 ) THEN
	            alen = alen + 1
	            line_buff(alen:alen) = tab_or_comma
	         ENDIF
 330	      CONTINUE
	
 331	      CALL SPLIT_LIST(pttmode_explct, list_lun,
     .				line_buff(:alen), alen )

	   ENDIF  ! skipped if heading turned off

* As for non-comma or -tab listings, turn off these labels if /norow was given

	   DO 355 n = lo6, hi6, del6
* ... label the block of blocks (6th dimension)
	      IF ( ndim .GE. 6 .AND. valid(6) .AND. 
     .               (.NOT.norow) ) THEN
	         CALL POS_LAB( l, cx, perm(6),
     .			    ndig(perm(6)), line_buff, alen )
	         CALL SPLIT_LIST(pttmode_explct, list_lun,
     .			' ---- '//line_buff(:alen), 6+alen )
	      ENDIF
	   DO 354 m = lo5, hi5, del5
* ... label the block of blocks (5th dimension)
	      IF ( ndim .GE. 5 .AND. valid(5) .AND. 
     .               (.NOT.norow) ) THEN
	         CALL POS_LAB( l, cx, perm(5),
     .			    ndig(perm(5)), line_buff, alen )
	         CALL SPLIT_LIST(pttmode_explct, list_lun,
     .			' ---- '//line_buff(:alen), 6+alen )
	      ENDIF
	   DO 353 l = lo4, hi4, del4
* ... label the block of blocks (4th dimension)
	      IF ( ndim .GE. 4 .AND. valid(4) .AND. 
     .               (.NOT.norow) ) THEN
	         CALL POS_LAB( l, cx, perm(4),
     .			    ndig(perm(4)), line_buff, alen )
	         CALL SPLIT_LIST(pttmode_explct, list_lun,
     .			' ---- '//line_buff(:alen), 6+alen )
	      ENDIF
	   DO 352 k = lo3, hi3, del3
* ... label a single block (3rd dimension)
	      IF ( ndim .GE. 3 .AND. valid(3) .AND. 
     .               (.NOT.norow) ) THEN
	         CALL POS_LAB( k, cx, perm(3),
     .			       ndig(perm(3)), line_buff, alen )
	         CALL SPLIT_LIST(pttmode_explct, list_lun,
     .			' ---- '//line_buff(:alen), 6+alen )
	      ENDIF
	   DO 351 j = lo2, hi2, del2
	      IF ( interrupted ) RETURN
* ...  create label for this row
	         IF ( ndim .GE. 2 .AND. valid(2) .AND. 
     .               (.NOT.norow) ) THEN
	            CALL TRANSLATE_TO_WORLD
     .		      ( TM_WORLD( j, grid, col_ax, box_middle ),
     .			col_ax, grid, ndig(col_ax), line_buff(:25) )
		    alen = TM_LENSTR1(line_buff(:25))
	         ELSE	! 1D row-only 
	            line_buff(1:5) = ''	
	            alen = 0
	         ENDIF
* ...  and append the values for this row
	         DO 340 i = lo1, hi1, del1
	            IF ( itsa_string ) THEN
	               IF (alen.GT.0) THEN
		          line_buff = line_buff(:alen)//tab_or_COMMA//
     .			GET_STRING_ELEMENT(
     .			m1lox,m1hix,m1loy,m1hiy,m1loz,m1hiz,m1lot,m1hit,
     .			m1loe,m1hie,m1lof,m1hif,grid_data,i,j,k,l,m,n,
     .			max_line_len,vlen )
	               ELSE 
		          line_buff = GET_STRING_ELEMENT(
     .			m1lox,m1hix,m1loy,m1hiy,m1loz,m1hiz,m1lot,m1hit,
     .			m1loe,m1hie,m1lof,m1hif,grid_data,i,j,k,l,m,n,
     .			max_line_len,vlen )
	               ENDIF

	            ELSE
                       fmtline = 
     .			TM_FMT(grid_data(i,j,k,l,m,n),prec_digits,16,vlen)
	               IF (alen.GT.0) THEN
		          line_buff = line_buff(:alen)//tab_or_COMMA//fmtline
	               ELSE
		          line_buff = fmtline
	               ENDIF
	            ENDIF
	            alen = alen+vlen+1
	            IF ( alen .GT. max_line_len ) THEN
	               line_buff(max_line_len-18:max_line_len) = 
     .			'** line too long **'
	               alen = max_line_len
	               GOTO 341
	            ENDIF
 340	         CONTINUE
 341	         CALL SPLIT_LIST(pttmode_explct, list_lun,
     .				line_buff(:alen), alen )

 351	   CONTINUE
 352	   CONTINUE
 353	   CONTINUE
 354	   CONTINUE
 355	   CONTINUE
* done - go home
	   RETURN
	ENDIF

* * * * PREPARE FORMATS ADAPTED TO THE DATA AND AXIS RANGES
* ... row labelling
	IF ( ndim .GE. 2 .AND. valid(2) ) THEN
	   CALL ROW_COORD_FMT( col_ax, cx, lo2, hi2, del2,
     .			       row_ss_fmt, col_dec, row_wld, cleft )
	ELSE
	   cleft = 7
	   line_buff(:cleft) = ' '		! just 1 value to write
	ENDIF
        IF (norow) cleft = 0

	usable_line_len = line_length - cleft
	max_fields = usable_line_len / min_field_width
* ... column labelling - how many decimal places for coordinate label ?
	IF ( ndim .GE. 1 .AND. valid(1) ) THEN
	   small = +1.E33
	   DO 404 i = lo1, hi1, del1
	      small = MIN( small, BOX_SIZE( i, grid, row_ax ) )
 404	   CONTINUE	      
	   CALL GET_PREC_DIGITS( cx_lo_ww(row_ax,cx),
     .				 cx_hi_ww(row_ax,cx),
     .				 small, places, row_dec )
	   row_dec = row_dec + 1		! add 1 more, arbitrarily
c	   IF ( GEOG_LABEL(col_ax,grid) ) row_dec = MAX(row_dec, ax_dec_pt(col_ax))
	ENDIF

* ... determine numeric format to fit nicely on the line
* ... first find largest value to be printed
! bug here since the mem res data limits may not match the cx limits 8/00 *sh*
	IF ( itsa_string ) THEN
	   nleft = GET_MAX_C_STRING_LEN(mr_c_pointer(mr), MGRID_SIZE(mr))
	   all_bad = .FALSE.
	ELSE
	   CALL MINMAX(grid_data, MGRID_SIZE(mr), bad_data, small, big, nok)
	   all_bad = nok .EQ. 0

* . . . how many digits right and left of decimal point ?
 	   biggest = MAX( ABS( small ), big )
	   CALL GET_SIG_DIGITS( biggest, ABS(prec_digits), nleft, nright )
	   nleft = MAX( 1, nleft )			! at least "0" to left
           IF ( prec_digits .GT. 0 ) THEN
*             positive means user is giving the total number of digits
              ndigits = prec_digits
           ELSE
*              zero or negative means user is giving nright; use nleft from above 
*              (independent of prec_digits) and compute the total number of digits
               nright = ABS(prec_digits)
               ndigits = nleft + nright
           ENDIF
	ENDIF

* . . . unreasonable line length ?
 410	IF ( ndim .EQ. 0 ) THEN
	   num_fields = 1
	ELSE
	   num_fields = ABS( (hi1-lo1) / del1  + 1 )
	   IF ( num_fields .GT. max_fields ) line_too_long = .TRUE.
* . . . shorten the line to something that will fit
	   IF ( line_too_long ) THEN
	      num_fld_est	= usable_line_len / width
	      temp		= ABS(hi1-lo1) / num_fld_est + 1.
	      del1		= SIGN( temp, del1 )		! for del1<0
	      num_fields 	= ABS( (hi1-lo1) / del1 + 1 )
	      del_col_changed	= .TRUE.
	   ENDIF
	ENDIF

* . . . determine FORTRAN format for listing
	IF ( all_bad ) THEN
	   width = min_field_width
	   WRITE ( bad_data_form, 3002 ) min_field_width
 3002	FORMAT ( '(A',I2,')' )	
	ELSE
	   IF ( itsa_string ) THEN
	      width = nleft + 2
	   ELSE
	      CALL VAR_DATA_FMT( nleft, nright, ndigits, min_field_width,
     .			      width, good_data_form, bad_data_form )
	   ENDIF
	ENDIF

* . . . will it fit ? - if not go back and try again with a shorter line
	dlen = cleft + width*num_fields
* . . . *kob* 9/03 - we do not want to set line_to_long to true for strings -
*                    results in a hang because dlen doesn't get changed or 
*                    shortened and so this will always be true for long strings
*		     and we will be sent back to 410 ad naseum
	IF ( dlen .GT. line_length .AND. .NOT. itsa_string ) THEN
	   line_too_long = .TRUE.
	   GOTO 410
	ENDIF
	
* . . . notify operator if were skipping some data
	IF ( del_col_changed ) THEN
	   WRITE ( line_buff, 3005 ) del1
 3005	FORMAT ( 5X,' ... listing every',I4,'th point' )
	   CALL SPLIT_LIST(pttmode_explct, list_lun, line_buff, 35 )
	ENDIF

* . . . write a heading line of coordinate values and subscripts ?
	IF ( heading .AND. ndim.GE.1 .AND. valid(1) ) THEN
	   line_buff(1:line_length) = ' '
	   w0 = cleft + 2	! 2 col arbitrary shift to align
	   DO 420 i = lo1, hi1, del1
	      val8 = TM_WORLD( i, grid, row_ax, box_middle )
	      first = FIELD_WIDTH(val8, grid, row_ax, row_dec, numbers)
	      first = 1 + MAX( 0, (width-first)/2 )	! centered
	      CALL TRANSLATE_TO_WORLD
     .		      ( val8,
     .			row_ax,
     .			grid,
     .			row_dec,
     .			line_buff(w0+first:w0+width) )
	      w0 = w0 + width 
 420	   CONTINUE
	   CALL SPLIT_LIST(pttmode_explct, list_lun,
     .				' '//line_buff(:w0), w0+1 )

* . . .  then subscript values ...
           llen = MAX(1, cleft+1-(width-3)/2)
	   WRITE ( col_head_format, 3020 ) llen, num_fields,  width
	   WRITE ( line_buff, col_head_format ) ( i,i = lo1,hi1,del1 ) 
 3020	   FORMAT ( '(',I2,'X,',I5,'I',I2,')' )	! eg. "( 8X, 10I 6)"
	   CALL SPLIT_LIST(pttmode_explct, list_lun, line_buff, 0 )
	ENDIF

* . . . list column of latitudes backwards (north to south)
	IF ( col_ax.EQ.Y_dim .AND. GEOG_LABEL(y_dim,grid) ) THEN
	   del2 = -1
	   temp = lo2
	   lo2  = hi2
	   hi2  = temp
	ENDIF

* * * * *  WRITE FORMATTED DATA LINES * * * * *
* each line has format: "coord/sub: data1 data2 data3 ..."
* for a 2D field a single block of such lines is printed
* for a 3D field each block is separated by an identifier of the 3rd axis
* for a 4D field each block of blocks is further separated and identified

	DO 550 n = lo6, hi6, del6
* label the block of blocks (6th dimension)
	   IF ( ndim .GE. 6 .AND. valid(6) ) THEN
	      CALL POS_LAB( n, cx, perm(6),
     .			    ax_dec_pt(perm(6)), line_buff, alen )
	      CALL SPLIT_LIST(pttmode_explct, list_lun,
     .			' ---- '//line_buff(:alen), 6+alen )
	   ENDIF

	DO 540 m = lo5, hi5, del5
* label the block of blocks (5th dimension)
	   IF ( ndim .GE. 5 .AND. valid(5) ) THEN
	      CALL POS_LAB( m, cx, perm(5),
     .			    ax_dec_pt(perm(5)), line_buff, alen )
	      CALL SPLIT_LIST(pttmode_explct, list_lun,
     .			' ---- '//line_buff(:alen), 6+alen )
	   ENDIF

	DO 530 l = lo4, hi4, del4
* label the block of blocks (4th dimension)
	   IF ( ndim .GE. 4 .AND. valid(4) ) THEN
	      CALL POS_LAB( l, cx, perm(4),
     .			    ax_dec_pt(perm(4)), line_buff, alen )
	      CALL SPLIT_LIST(pttmode_explct, list_lun,
     .			' ---- '//line_buff(:alen), 6+alen )
	   ENDIF

	   DO 520 k = lo3, hi3, del3
*    label a single block (3rd dimension)
	      IF ( ndim .GE. 3 .AND. valid(3) ) THEN
	         CALL POS_LAB( k, cx, perm(3),
     .			       ax_dec_pt(perm(3)), line_buff, alen )
	         CALL SPLIT_LIST(pttmode_explct, list_lun,
     .			' ---- '//line_buff(:alen), 6+alen )
	      ENDIF

	      DO 510 j = lo2, hi2, del2
*       create carefully sized label for this row
	         IF ( ndim .GE. 2 .AND. valid(2) .AND. 
     .               (.NOT.norow) ) THEN
                    val8 = TM_WORLD( j, grid, col_ax, box_middle )
	            CALL TRANSLATE_TO_WORLD
     .		      ( val8,
     .			col_ax, grid, col_dec, line_buff(:row_wld) )
	            WRITE ( line_buff(row_wld+1:cleft), row_ss_fmt ) j
	         ENDIF
	         ifield = 0

	         DO 500 i = lo1, hi1, del1
* put values into the output line buffer
	         ifield = ifield + 1
	         iend   = cleft + ifield*width
	         istart = iend - width + 1
	         IF (itsa_string) THEN
	           line_buff(istart:istart) = '"'
	           line_buff(istart+1:iend) = GET_STRING_ELEMENT(
     .			m1lox,m1hix,m1loy,m1hiy,m1loz,m1hiz,m1lot,m1hit,
     .			m1loe,m1hie,m1lof,m1hif,grid_data,i,j,k,l,m,n,
     .			width-2,slen)
	           IF (slen .GT. width-2) THEN
	              line_buff(iend-3:iend) = '..."'
	           ELSE
	              line_buff(istart+slen+1:istart+slen+1) = '"'
	           ENDIF
	         ELSE
	           value = grid_data(i,j,k,l,m,n)
	           IF ( value .NE. bad_data ) THEN
	              WRITE ( line_buff(istart:iend), good_data_form, err=5030 ) value
	           ELSE
	              WRITE ( line_buff(istart:iend),  bad_data_form, err=5040 ) '....'
	           ENDIF
	         ENDIF
 500	         CONTINUE

	         IF ( interrupted ) RETURN
	         CALL SPLIT_LIST(pttmode_explct, list_lun,
     .					' '//line_buff(:dlen),dlen+1) 
 510	      CONTINUE
 520	   CONTINUE
 530	CONTINUE
 540	CONTINUE
 550	CONTINUE

* successful completion
	RETURN

* error exits
 5000	RETURN
 5010	CALL ERRMSG( ferr_erreq, status, list_format, *5000 )

 5020	CALL ERRMSG( ferr_invalid_command, status,
     .		'Integer format allows one value per line on output '//
     .		list_format(:TM_LENSTR1(list_format)) ,
     .		*5000 )
 5030	write (buff, *) value
	CALL ERRMSG( ferr_nomessge, status,
     .		'Unknown error writing data ' //
     .		buff(:TM_LENSTR1(buff)) ,
     .		*5000 )
 5040	CALL ERRMSG( ferr_nomessge, status,
     .		'Unknown error writing missing-data line ' ,
     .		*5000 )

	END
