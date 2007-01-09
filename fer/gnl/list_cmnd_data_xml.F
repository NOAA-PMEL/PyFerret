        SUBROUTINE LIST_CMND_DATA_XML( grid_data, mr, cx,
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
* list a gridded variable stored in memory - 0 to 4 dimensions
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
*	           - expanded and reorganized the LIST heading
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
	INTEGER		mr, cx, perm(4), recsofar, status
	REAL grid_data(m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit),
     .       ax_buff(*)
        CHARACTER*(*) dods_file
	
* local variable declarations:
	LOGICAL		GEOG_LABEL, TM_DFPEQ, 
     .			line_too_long, del_col_changed, all_bad,
     .			valid(4), need_doc(4), span(4), do_unform,
     .			do_stream, do_comma_del, do_tab_del, itsa_string,
     .			fortran_selfdoc, do_auto_fmt, do_dods, norow
	INTEGER		TM_LENSTR, TM_LENSTR1, CX_DIM_LEN,
     .			FIELD_WIDTH, MGRID_SIZE,
     .			GET_MAX_C_STRING_LEN, TM_GET_CALENDAR_ID,
     .			width, w0, grid, dlen, prec_digits,
     .			ndim, lo(4), hi(4), del(4),
     .			i, j, k, l, npts, line_length, dset,
     .			temp, num_fld_est, nleft, nright, alen, num_fields,
     .			idim, vlen, istart, iend, ifield,
     .			cleft, usable_line_len, max_fields, row_ax,
     .			col_ax, places, col_dec, row_dec, row_wld, 
     .			numbers, first, slen, llen, axis, lp, cal_id,
     .                  wkblk, ndig(4)
	CHARACTER*255	VAR_TITLE, VAR_TITLE_MOD, PAREN_SURROUND, vname_buff
* V500: *kob* 3/99 - up VAR_UNITS from 32 to 64 
	CHARACTER	AX_TITLE*32, VAR_UNITS*64, LEFINT*8, TM_FMT*16,
     .			GET_STRING_ELEMENT*512,
     .			line_buff*10000, entry_value*30, row_value*30,
     .                  k_value*30, l_value*30,
     .			good_data_form*10, row_ss_fmt*14, tab_or_comma*1,
     .			bad_data_form*10, col_head_format*32, cal_name*32
	REAL		BOX_SIZE, big, small, biggest, bad_data, value
	REAL*8		TM_WORLD, val8, val_last, diff_min, diff_ave, count

#ifdef DEC_FORTRAN
	INTEGER		ios
#endif

* convenience equivalence:
	INTEGER		lo1, lo2, lo3, lo4, hi1, hi2, hi3, hi4,
     .			del1, del2, del3, del4
	EQUIVALENCE	(lo1, lo(1)), (lo2, lo(2)), (lo3,lo(3)),  (lo4,lo(4)),
     .			(hi1, hi(1)), (hi2, hi(2)), (hi3,hi(3)),  (hi4,hi(4)),
     .			(del1,del(1)),(del2,del(2)),(del3,del(3)),(del4,del(4))

* local parameter declarations
	INTEGER		slash_precision, slash_norow
	PARAMETER     (	slash_precision = 17+1,
     .                  slash_norow   = 32+1 )

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
        norow = qual_given(slash_norow) .GT. 0

* determine precision for outputting values
	lp = qual_given( slash_precision ) 
	IF ( lp .GT. 0 ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(lp):qual_end(lp)),
     .                     value, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
	   prec_digits = MAX(1.,MIN(value, 16.))	! 1 to 16	   
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
	DO 10 ndim = 4, 1, -1
	   IF ( CX_DIM_LEN( perm(ndim), cx ) .GT. 1 ) GOTO 20
 10	CONTINUE
	ndim = 0
 20	DO 30 idim = 1, 4
	   span(idim) = CX_DIM_LEN( perm(idim), cx ) .GT. 1
	   valid(idim) = cx_lo_ss( cx, perm(idim) ) .NE. unspecified_int4
 30	CONTINUE
* * * * * * WRITE INTRODUCTORY LINES * * * * *
	IF ( do_unform .OR. do_stream .OR. do_dods .OR.
     .       .NOT.heading ) GOTO 200
* ... first the variable name
	vname_buff   = VAR_TITLE( cx )
	vlen   = TM_LENSTR1( vname_buff )
	vname_buff   = vname_buff(:vlen)//' '//
     .				PAREN_SURROUND(VAR_UNITS(cx), alen)
        slen=TM_LENSTR1(vname_buff)
	CALL SPLIT_LIST(pttmode_explct, list_lun,
     .		'<variable>'//vname_buff(:slen)
     .          //'</variable>', 0)

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
             slen=TM_LENSTR1(line_buff)
	     CALL SPLIT_LIST(pttmode_explct, list_lun,
     .		  '<dataset>'//line_buff(:slen)//'</dataset>',0)
	   ENDIF
* ... find split point between filename and path
	   slen = TM_LENSTR1(ds_des_name(dset))
	   DO i = slen, 1, -1
	      IF ( ds_des_name(dset)(i:i) .EQ. '/' ) GOTO 34
	   ENDDO
	   i = 0
* ... filename
 34	   CALL SPLIT_LIST(pttmode_explct, list_lun,
     .	        '<filename>'//
     .		ds_des_name(dset)(i+1:slen)//'</filename>', 0)
* ... DODS URL BASE or ABSOLUTE PATH
* *kob* v553 - CYGWIN paths are always preceded by '.'
#ifdef __CYGWIN__
	   IF (i.GT.0 .AND.  ds_des_name(dset)(2:2).EQ.'\')
#else
	   IF (i.GT.0 .AND.  ds_des_name(dset)(1:1).NE.'.')
#endif
     .			CALL SPLIT_LIST(pttmode_explct, list_lun,
     .			'<filepath>'//
     .			ds_des_name(dset)(:i)//'</filepath>',0)
	ENDIF

* ... BAD FLAG: xxx
	IF (.NOT.fortran_selfdoc) THEN
	   IF (.NOT.do_auto_fmt) THEN
	     line_buff = ' '
	     WRITE ( line_buff, '(1PG14.7)' ) cx_bad_data(cx)
             slen=TM_LENSTR1(line_buff)
	     CALL SPLIT_LIST( pttmode_explct, list_lun,
     .			'<badflag>'//
     .		        line_buff(:slen)//'</badflag>', 0)
	   ENDIF
* ... SUBSET:   e.g. "Nx by Ny (longitude-latitude)"
	   line_buff = '<subset>'
	   l = 8
	   DO idim = 1, 4
	      IF ( span(idim) ) THEN
	         i = perm(idim)
	         j = CX_DIM_LEN ( i, cx )
	         line_buff = line_buff(:l)//LEFINT( j, vlen )
	         l = l + vlen
	         line_buff = line_buff(:l)//' by '
	         l = l + 4
	      ENDIF
	   ENDDO
	   IF ( l.GT.8) THEN
	      line_buff(l-3:l+7) = ' points (  '
	      l = l + 5
	      DO idim = 1, 4
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
     .			      line_buff(:l)//'</subset>', 0)
	   ENDIF
	ENDIF

* ... which axes require labels
	DO 35 idim = 1, 4
 35	need_doc(idim) = .TRUE.
	IF ( .NOT.(do_unform.OR.do_dods.OR.list_format_given) ) THEN
	   DO 36 idim = 1, ndim
 36	   need_doc(idim) = CX_DIM_LEN( perm(idim), cx ) .EQ. 1
	ENDIF

* ... if not indicated in time label show a nonstandard calendar name
	cal_name = line_cal_name( grid_line(t_dim, grid) )
	cal_id = TM_GET_CALENDAR_ID (cal_name)
	DO i = 1, 4
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
	DO 60 i = 1, 4
	   idim = perm(i)
	   IF ( need_doc(i) ) THEN
	      IF ( cx_lo_ww( idim,cx ) .NE. unspecified_val8 ) THEN 
	         CALL ASCII_LIMITS( cx, idim, line_buff, vlen )
	         col_head_format = AX_TITLE( idim, grid, alen )
	         IF (alen .LT. 9) alen = 9
!	         CALL SPLIT_LIST( pttmode_explct, list_lun,
!     .			'             '//col_head_format( :alen )//': '
!     .			//line_buff(1:vlen), alen+vlen+15)

                 CALL SPLIT_LIST( pttmode_explct, list_lun,
     .                 '<'//col_head_format( :alen )//'>'
     .                 //line_buff(1:vlen)//
     .                 '</'//col_head_format( :alen)//'>', 0)

	      ENDIF
	   ENDIF
 60	CONTINUE

* * * * * DETERMINE LIST LIMITS * * *
 200	DO 210 i = 1, 4
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
	   DO 220 idim = 1, 4
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
	   DO 230 idim = 1, 4
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
	   DO 250 idim = 1, 4
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

	   DO 300 l = lo4, hi4, del4
	   DO 300 k = lo3, hi3, del3
	   DO 300 j = lo2, hi2, del2
	      IF ( interrupted ) RETURN
! ----- start string handler -------------
	      IF (itsa_string) THEN
	         llen = 0
	         DO 270 i = lo1, hi1, del1
	            line_buff(llen+1:) = GET_STRING_ELEMENT(
     .			m1lox,m1hix,m1loy,m1hiy,m1loz,m1hiz,m1lot,m1hit,
     .			grid_data,i,j,k,l,max_line_len,slen )
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
     .			grid_data,i,j,k,l,max_line_len,slen )
	               WRITE (list_lun,list_format, ERR=5010)
     .							line_buff(:slen)
 272		    CONTINUE
	         ENDIF
	         GOTO 300
	      ENDIF
! ----- end string handler -------------
	      IF ( do_unform ) THEN
	         WRITE (list_lun, ERR=5010)
     .			( grid_data(i,j,k,l),i = lo1, hi1, del1 )

	      ELSEIF ( do_stream ) THEN				! 12/94
	         DO 298 i = lo1, hi1, del1
	            recsofar = recsofar + 1
	            WRITE (list_lun, REC=recsofar, ERR=5010)
     .						grid_data(i,j,k,l)
 298		 CONTINUE
	      ELSE
! Note: FORMATTED output cannot be passed through SPLIT_AXIS because a
!	single WRITE may imply many records of output
#ifdef DEC_FORTRAN
* on DEC compilers a field too small for the value generates an ERR= branch
	         WRITE (list_lun,list_format, IOSTAT=ios)
     .			( grid_data(i,j,k,l),i = lo1, hi1, del1 )
	         IF ( ios.NE.0 .AND. ios.NE.63 ) GOTO 5010  ! 63 = fmt conv err
#else
	         WRITE (list_lun,list_format, ERR=5010)
     .			( grid_data(i,j,k,l),i = lo1, hi1, del1 )
#endif
	      ENDIF
 300	   CONTINUE

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
	   line_buff(1:line_length) = 'coordinates'//tab_or_comma
	   alen = 12

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

* Write coordinates

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
 330	   CONTINUE
 331	   CALL SPLIT_LIST(pttmode_explct, list_lun,
     .				line_buff(:alen), alen )
	   DO 350 l = lo4, hi4, del4
* ... label the block of blocks (4th dimension)
	      IF ( ndim .GE. 4 .AND. valid(4) ) THEN
	         CALL POS_LAB( l, cx, perm(4),
     .			    ndig(perm(4)), line_buff, alen )
	         CALL SPLIT_LIST(pttmode_explct, list_lun,
     .			' ---- '//line_buff(:alen), 6+alen )
	      ENDIF
	   DO 350 k = lo3, hi3, del3
* ... label a single block (3rd dimension)
	      IF ( ndim .GE. 3 .AND. valid(3) ) THEN
	         CALL POS_LAB( k, cx, perm(3),
     .			       ndig(perm(3)), line_buff, alen )
	         CALL SPLIT_LIST(pttmode_explct, list_lun,
     .			' ---- '//line_buff(:alen), 6+alen )
	      ENDIF
	   DO 350 j = lo2, hi2, del2
	      IF ( interrupted ) RETURN
* ...  create label for this row
	         IF ( ndim .GE. 2 .AND. valid(2) ) THEN
	            CALL TRANSLATE_TO_WORLD
     .		      ( TM_WORLD( j, grid, col_ax, box_middle ),
     .			col_ax, grid, ndig(col_ax), line_buff(:16) )
	            alen = TM_LENSTR1(line_buff(:16))
	         ELSE	! 1D row-only (will this ever happen?)
	            line_buff(1:5) = 'row 1'	
	            alen = 5
	         ENDIF
* ...  and append the values for this row
	         DO 340 i = lo1, hi1, del1
	            IF ( itsa_string ) THEN
	               line_buff = line_buff(:alen)//tab_or_COMMA//
     .			GET_STRING_ELEMENT(
     .			m1lox,m1hix,m1loy,m1hiy,m1loz,m1hiz,m1lot,m1hit,
     .			grid_data,i,j,k,l,max_line_len,vlen )
	            ELSE
	               line_buff = line_buff(:alen)//tab_or_COMMA//
     .			TM_FMT(grid_data(i,j,k,l),prec_digits,16,vlen)
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
 350	   CONTINUE
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
	ENDIF

* ... determine numeric format to fit nicely on the line
* ... first find largest value to be printed
! bug here since the mem res data limits may not match the cx limits 8/00 *sh*
	IF ( itsa_string ) THEN
	   nleft = GET_MAX_C_STRING_LEN(mr_c_pointer(mr), MGRID_SIZE(mr))
	   all_bad = .FALSE.
	ELSE
	   CALL MINMAX(grid_data, MGRID_SIZE(mr), bad_data, small, big)
	   all_bad = small .EQ. arbitrary_large_val4

* . . . how many digits right and left of decimal point ?
 	   biggest = MAX( ABS( small ), big )
	   CALL GET_SIG_DIGITS( biggest, prec_digits, nleft, nright )
	   nleft = MAX( 1, nleft )			! at least "0" to left
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
	      CALL VAR_DATA_FMT( nleft, nright, prec_digits, min_field_width,
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
	   WRITE ( col_head_format, 3020 ) cleft+1-(width-3)/2,
     .					      num_fields,  width
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

	DO 530 l = lo4, hi4, del4

           CALL SPLIT_LIST(pttmode_explct, list_lun,
     .            '<tgroup>',0)                 


* label the block of blocks (4th dimension)
	   IF ( ndim .GE. 4 .AND. valid(4) ) THEN
!           IF (valid(4)) THEN
	      CALL POS_LAB( l, cx, perm(4),
     .			    ax_dec_pt(perm(4)), line_buff, alen )
	      CALL SPLIT_LIST(pttmode_explct, list_lun,
!     .			' ---- '//line_buff(:alen), 6+alen )
     .             '  <4D value=\''//line_buff(:alen)//'\'>', 0)
	   ENDIF

	   DO 520 k = lo3, hi3, del3

*    label a single block (3rd dimension)
	      IF ( ndim .GE. 3 .AND. valid(3) ) THEN
!	         CALL POS_LAB( k, cx, perm(3),
!     .			       ax_dec_pt(perm(3)), line_buff, alen )
                 CALL TRANSLATE_TO_WORLD
     .                ( TM_WORLD( k, grid, perm(3), box_middle ),
     .                  perm(3), grid, ax_dec_pt(perm(3)), k_value )
                    slen= TM_LENSTR1(k_value)

	         CALL SPLIT_LIST(pttmode_explct, list_lun,
     .            '    <3D value=\''//k_value(:slen)//'\'>',0)                 
	      ENDIF

	      DO 510 j = lo2, hi2, del2
*       create carefully sized label for this row
	         IF ( ndim .GE. 2 .AND. valid(2) .AND. 
     .               (.NOT.norow) ) THEN

	            CALL TRANSLATE_TO_WORLD
     .		      ( TM_WORLD( j, grid, col_ax, box_middle ),
!     .			col_ax, grid, col_dec, line_buff(:row_wld) )
     .                  col_ax, grid, col_dec, row_value )
!	            WRITE ( line_buff(row_wld+1:cleft), row_ss_fmt ) j
                    slen= TM_LENSTR1(row_value)
                    CALL SPLIT_LIST(pttmode_explct, list_lun,
     .                  '       <row value=\''//row_value(:slen)
     .                  //'\'>', 0)

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
     .			grid_data,i,j,k,l,width-2,slen)
	           IF (slen .GT. width-2) THEN
	              line_buff(iend-3:iend) = '..."'
	           ELSE
	              line_buff(istart+slen+1:istart+slen+1) = '"'
	           ENDIF
	         ELSE
	           value = grid_data(i,j,k,l)
	           IF ( value .NE. bad_data ) THEN
	              WRITE ( entry_value, good_data_form ) value
	           ELSE
	              WRITE ( entry_value,  bad_data_form ) '....'
	           ENDIF
	         ENDIF
                 slen=TM_LENSTR1(entry_value)
                 CALL SPLIT_LIST(pttmode_explct, list_lun,
     .           '         <entry>'//entry_value(3:slen)//'</entry>', 0)

 500	         CONTINUE

	         IF ( interrupted ) RETURN
!	         CALL SPLIT_LIST(pttmode_explct, list_lun,
!     .			'         <entry>'//line_buff(istart:iend)//'</entry>', 0)

                 IF ( ndim .GE. 2 .AND. valid(2) .AND.
     .               (.NOT.norow) ) THEN

                     CALL SPLIT_LIST(pttmode_explct, list_lun,
     .                '       </row>', 0)
                 ENDIF

 510	      CONTINUE
              IF ( ndim .GE. 3 .AND. valid(3) ) THEN
                  CALL SPLIT_LIST(pttmode_explct, list_lun,
     .             '    </3D>', 0)
              ENDIF

 520	   CONTINUE
           IF ( ndim .GE. 4 .AND. valid(4) ) THEN
               CALL SPLIT_LIST(pttmode_explct, list_lun,
     .             '  </4D>', 0)
           ENDIF

           CALL SPLIT_LIST(pttmode_explct, list_lun,
     .            '</tgroup>',0)                 

 530	CONTINUE

* successful completion
	RETURN

* error exits
 5000	RETURN
 5010	CALL ERRMSG( ferr_erreq, status, list_format, *5000 )
	END
