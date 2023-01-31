	SUBROUTINE LIST_MULTI_SUB( mr_list,cx_list,heading,
     .				  perm,ws,xact_range, has_size, bad,
     .				  pt_blk, size, msize, width, out,
     .				  status)

*
*
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
* list several gridded variables with multiple variables per record

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:   7/5/90 - based on LIST_CMND_DATA
* V230:  6/18/92 - 2 bug fixes for listings with ambiguous coords
*       list/k=2/l=5/x=130e:180/y=0:20/d=gtsa056 temp,salt,u,v[k=1]
*                - use COMMON variable list_format_type
*        7/22/92 - identify "column 1", "column 2", ... in header (bug 7/31)
*        11/4/92 - expanded variable name length
* V300:  4/29/93 - fixed subscript bug for xact lists longer than mem_blk_size
* V301: 11/1/93 - change to WRITE(ERR=...) processing for DEC compilers
* V312: 5/94 - array "memory" as a calling argument
* V313: 8/14/94 *kob* IBM port - variable declaration for ss_save changed from
*				 integer to logical
*	31/10/94 *sh* and size of row_ss_fmt increased due to ROW_COORD_FMT chg
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
* V500 *kob* 3/99- up VAR_CODE and vcode decl's from 8 to 64 chars
* V530  8/00 - added support for string variables
* V552 *acm* 4/03- up VAR_CODE and vcode decl's from 64 to 128 chars
* V581: *acm*/3/05 - increase length of strings for dataset title to 1024
* V600: *acm* 8/05 - make multi-variable tab and comma-delimited listings of
*                    multiple variables like free-format listings, rather than 
*                    like LIST/SINGLY. (bug 1273)
* V600: *acm* 8/05 - Add new qualifier LIST/NOROWHEAD to remove labels that
*                    start each row labelling the coordinate values. (bug 1273)
* V600: *acm* 8/05 - fix bug 1313; reinitialize list_buffer after heading of 
*                    variable names on columns, before listing data
* V602: *acm 11/06 - fix bug 1451 when list/form=comma or /format=tab,
*                    and there is missing data, output a comma or tab
*                    placeholder for the missing values'
* V603: *acm* 6/07 - Fix bug 1388. LIST/PREC= did not work on multi-variable 
*                    listings but SET LIST/PREC=; LIST  did. The code to get
*                    the value of /PREC= was inside an IF block only executed for
*                    tab and comma formatting, and there was also an inconsistency
*                    in how the two flags were used: prec_digits(for LIST/PREC=) and
*                    list_digits(setting from SET LIST/PREC=)
* V610: *acm* 2/10 - Fix bug 1698. For tab and comma-delimited files,
*                    list out the missing data values. The point of these
*                    is that they'll be read into some other software e.g.
*                    spreadsheets, so the user wants the missing flags in there.
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* V691+ *acm* 9/14 Ticket 2199: for string variables don't list a BAD FLAG in the header.
* V702: *sh*  2/17 Dynamic memory management (pt_blk no longer used)
* V740 *acm*  3/18 Issue 950: Allow (some) A formats for listing strings and numeric data together
*                  Also simplified I formats in the same way.
* V741 *acm*  4/18 Issue 1870: Check Format more closely when finding an I Format
* V751 6/19 *acm* TM_FMT rewrite: formatting of lon/lat coordinate strings
* V76  *acm*  5/20  More on Issue 1870
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret.parm'
	include 'errmsg.parm'
	include	'slash.parm'
	include 'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xprog_state.cmn'
	include 'xcontext.cmn'
	include 'xtext_info.cmn'
	include 'xinterrupt.cmn'
	include 'xrisc.cmn'          ! 12/94 SPLIT_LIST buffer

* local parameter declarations:
	LOGICAL		no_decims
	INTEGER		line_length
	PARAMETER     ( line_length  = 2048,
     .			no_decims = .FALSE. )

* calling argument declarations:
	LOGICAL	heading
	INTEGER	mr_list(num_uvars_in_cmnd), cx_list(num_uvars_in_cmnd),
     .		perm(nferdims), ws, status
* ... work arrays ...
	LOGICAL		xact_range(ws), has_size(nferdims,ws)
	INTEGER		pt_blk(ws), size(nferdims,ws), msize(ws), width(ws)
	REAL		out(ws), bad(ws)
* ss_save removed from integer declaration and put in logical decl.  *kob* 8/94	
* V500 *kob* 3/99- up VAR_CODE and vcode decl's to 64 chars and KEY_STRING
*                  to 128 and VAR_UNITS to 64
* V552 *acm* 4/03- up VAR_CODE and vcode decl's from 64 to 128 chars
* V74+ *acm* 10/18 routine minmax returns a new argument: count of valid data

* local variable declarations:
	LOGICAL		TM_HAS_STRING, TM_DIGIT, all_bad, do_unform, ss_save, 
     .                  do_tab_del, do_comma_del, norow, has_string, has_int
	INTEGER		TM_LENSTR1, CGRID_AXIS, GET_MAX_STRING_LEN,
     .			GET_OFFSET_C_STRING_LEN, STR_SAME,
     .			w0, w, data_len, nleft, nright, alen, blen,
     .			idim, vlen, cleft, col_ax, col_dec,
     .			cx, slen1, slen2, slen3, i, 
     .			ivar, point, iwd, dset, ax, new_ax,
     .			min_width, row_ww_wid, lp, prec_digits, ndigits, ww,
     .			lo(nferdims), hi(nferdims), del(nferdims), 
     .			pt(nferdims), valid_ax(nferdims), lloc, num_it, ival, nok
	CHARACTER*255	VAR_TITLE, VAR_TITLE_MOD, PAREN_SURROUND, vname_buff
	CHARACTER	KEY_STRING*128, AX_TITLE*32, VAR_UNITS*64, LEFINT*8,
     .			CX_DIM_STR*16, VAR_CODE*128, TM_FMT*48,
     .			buff1*16, buff2*1024, buff3*16, line_buff*2048,
     .			row_ss_fmt*14, bad_ax*5, bad_buff*32,
     .			col_head_format*32, sep*1, fmtstring*30
	REAL		big, small, biggest, value
	REAL*8		TM_WORLD
	CHARACTER	good_fmt(max_intrp)*10, bad_fmt(max_intrp)*10,
     .			vcode(max_intrp)*128, tab_or_comma*1, b1*1

* convenience equivalence:
	INTEGER		lo1, lo2, lo3, lo4, lo5, lo6, 
     .			hi1, hi2, hi3, hi4, hi5, hi6, 
     .			del1,del2,del3,del4,del5,del6, 
     .			i1, i2, i3, i4, i5, i6
	EQUIVALENCE	
     .  (i1, pt(1)) , (i2, pt(2)) , (i3, pt(3)),  (i4,pt(4)),   (i5, pt(5)),  (i6,pt(6)),
     .  (lo1, lo(1)), (lo2, lo(2)), (lo3,lo(3)),  (lo4,lo(4)),  (lo5,lo(5)),  (lo6,lo(6)),
     .  (hi1, hi(1)), (hi2, hi(2)), (hi3,hi(3)),  (hi4,hi(4)),  (hi5,hi(5)),  (hi6,hi(6)),
     .  (del1,del(1)),(del2,del(2)),(del3,del(3)),(del4,del(4)),(del5,del(5)),(del6,del(6))

* initialize

	status		= ferr_ok
        do_unform       = list_fmt_type .EQ. plist_unformatted
	risc_buff = ' '

        do_tab_del   = list_fmt_type .EQ. plist_tab_del
        do_comma_del = list_fmt_type .EQ. plist_comma_del

        norow = qual_given(slash_list_norow) .GT. 0

        IF ( do_comma_del .OR. do_tab_del ) THEN
* ... column headings for tab and comma-delimited output (9/95)
	   IF ( do_comma_del ) THEN
	      tab_or_comma = ','
	   ELSEIF( do_tab_del ) THEN
	      tab_or_comma = CHAR(9)
           ENDIF
              
	ENDIF

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


* Check format specifiers if given.  When there are strings or integers we
* require simple format specs only.

	IF ( list_format_given ) THEN

	   risc_buff = ' '  ! this will store the value of any string outputs
	   line_buff = ' '  ! this will store a list of the internal variables to write, using format

* If there are strings, the list must be simplified with each variable given its own format.

	   has_string = .FALSE.
	   DO  ivar = 1, num_uvars_in_cmnd
	      IF ( mr_type(mr_list(ivar)) .EQ. ptype_string ) has_string = .TRUE.
	   ENDDO
* 
	   slen1 = TM_LENSTR1(list_format)
	   CALL PARSE_COMMA_LIST (list_format, 2, slen1-1, max_item_list,
     .	          num_it, item_start, item_end, status)

	   IF (num_it.NE.num_uvars_in_cmnd .AND. has_string) GOTO 5050	
	   
	   has_int = TM_HAS_STRING (list_format, 'I')
	   IF (num_it.LT.num_uvars_in_cmnd .AND. has_int)  GOTO 5050	

* Make sure the I is a format spec, not within a string in the format  

	   IF (has_int) THEN
	      has_int = .FALSE.
	      DO i = 1, num_it
	         buff2 = list_format(item_start(i):item_end(i))
		 
		 ! Is this item from the fmt statement a quoted string?
		 i1 = item_start(i) - 1
		 IF (i1.GE.1) THEN         
		    b1 = list_format(i1:i1)
		    IF ( b1 .EQ. '"' .OR.  b1 .EQ. "'") CYCLE
		 ENDIF

* Not within a string, is the first letter an I? 
		 has_int = (STR_SAME(buff2(1:1), 'i') .EQ. 0)
		 IF (has_int) EXIT ! from loop
	      ENDDO
	   ENDIF

	   IF (num_it.NE.num_uvars_in_cmnd .AND. (has_string .OR. has_int)) GOTO 5050

	   IF (has_string .OR. has_int) THEN
           

* For some combinations of specifiers, say (A12, 1X, 2F8.1)  it may seem there's
*  the right number of specifiers, but more complex than we are going to parse.

	      DO i1 = 1, num_it
	         IF ( TM_HAS_STRING(list_format(item_start(i1):item_end(i1)), 'X') )  
     .                GOTO 5050
	         IF ( TM_DIGIT (list_format(item_start(i1):item_end(i1)) ) )  GOTO 5050
	      ENDDO

	   ENDIF
	ENDIF      

* determine list limits checking for conformable shapes of data to list
* ... for each axis each var must be same size or length 1
	DO 10 i = 1, nferdims
	   idim = perm(i)
	   lo(i) = unspecified_int4
	   hi(i) = unspecified_int4
	   del(i)= 1
	   DO 8 ivar = 1, num_uvars_in_cmnd
	      cx = cx_list(ivar)
	      IF ( cx_lo_ss(cx,idim) .NE. cx_hi_ss(cx,idim) ) THEN
	         IF ( lo(i) .EQ. unspecified_int4 ) THEN
	            lo(i) = cx_lo_ss( cx, idim )
	            hi(i) = cx_hi_ss( cx, idim )
	         ELSEIF ( lo(i) .NE. cx_lo_ss(cx,idim)
     .		     .OR. hi(i) .NE. cx_hi_ss(cx,idim) ) THEN
	            GOTO 5020	! not conformable shapes
	         ENDIF
	      ENDIF
 8	   CONTINUE
 10	CONTINUE

* compute lengths of each variables' axes
	DO 20 i = 1, nferdims
	   idim = perm(i)
	   DO 18 ivar = 1, num_uvars_in_cmnd
	      cx = cx_list(ivar)
	      has_size( i, ivar ) = cx_hi_ss(cx,idim) .GT. cx_lo_ss(cx,idim)
	      size( i, ivar ) = cx_hi_ss(cx,idim) - cx_lo_ss(cx,idim) + 1
 18	   CONTINUE
 20	CONTINUE

* ... and total size
	DO ivar = 1, num_uvars_in_cmnd
	   msize(ivar) = 1
	   DO idim = 1, nferdims	
	      msize(ivar) = msize(ivar) * size(idim,ivar)	
	   ENDDO
	ENDDO

* determine which axes have unambiguous coordinates (all variables that are
* not point locations on this axis must have the same coordinates)
* create a string (bad_ax) with a list of ambiguous axes and a list, valid_ax,
* of context pointers for axes with valid lists of coordinates
	sep = ' '
	bad_ax = ' '
	slen1 = 1	! length of bad axis string
	DO 30 i = 1, nferdims
	   idim = perm(i)
	   valid_ax(i) = 0	! assume ambiguous coordinates
	   ax = mnormal
	   DO 28 ivar = 1, num_uvars_in_cmnd
	      cx = cx_list(ivar)
	      new_ax = CGRID_AXIS( idim, cx )
	      IF ( new_ax .EQ. mnormal ) GOTO 28
* ... point location context needed if all vars have point locations
	      IF ( .NOT.has_size(i,ivar) ) THEN
	         IF ( ax .EQ. mnormal ) valid_ax(i) = -cx
	         GOTO 28
	      ENDIF
	      IF ( ax .EQ. mnormal ) THEN
	         ax = new_ax
	         valid_ax(i) = cx
	      ELSEIF ( ax .NE. new_ax
     .		.OR.   cx_lo_ss(cx,idim) .NE. lo(i) ) THEN
	         valid_ax(i) = 0	! ambiguous
	         bad_ax = bad_ax(:slen1)//sep//ww_dim_name(idim)
		 sep = ','
	         slen1 = MIN( slen1 + 2, 5 )  ! bug fix 6/92
	      ENDIF
 28	   CONTINUE
 30	CONTINUE

* ... warn of ambiguous coordinates if output is not to the screen
	IF ( slen1.GT.1 .AND. ttout_lun.NE.list_lun ) THEN
	   WRITE (risc_buff, 3010) bad_ax(2:slen1)
	   CALL SPLIT_LIST
     .		(pttmode_bull, ttout_lun, risc_buff, 0)
	   risc_buff = ' '
	ENDIF
	   
* (for optimization) which variables are full sized ?
	DO 40 ivar = 1, num_uvars_in_cmnd
	   cx = cx_list(ivar)
	   xact_range(ivar) = .TRUE.
	   DO 38 i = 1, nferdims
	      idim = perm(i)
	      IF ( lo(i) .EQ. hi(i) ) GOTO 38
	      IF ( cx_lo_ss(cx,idim) .NE. lo(i)
     .	      .OR. cx_hi_ss(cx,idim) .NE. hi(i) ) THEN
	         xact_range(ivar) = .FALSE.
	         GOTO 40
	      ENDIF
 38	   CONTINUE
 40	CONTINUE

* (for optimization) save stuff
	DO 50 ivar = 1, num_uvars_in_cmnd
	   cx = cx_list(ivar)
	   vcode(ivar)  = VAR_CODE( cx_category(cx), cx_variable(cx) )
	   bad(ivar)    = mr_bad_data(  mr_list(ivar) )
 50	CONTINUE

* * * * * * WRITE INTRODUCTORY LINES * * * * *
	IF ( do_unform .OR. .NOT.heading ) GOTO 200

* ... begin with warning if coordinates are ambiguous
	IF ( slen1 .GT. 1 ) THEN
	   WRITE ( risc_buff, 3010 ) bad_ax(2:slen1)
	   CALL SPLIT_LIST
     .		(pttmode_bull, list_lun, risc_buff, 0)
	   risc_buff = ' '
	ENDIF
 3010	FORMAT (' WARNING: Listed variables have ambiguous ',
     .            'coordinates on axes:',A)

* ... summarize what the contexts have in common
	CALL GRP_CX_DOC( cx_list, num_uvars_in_cmnd, cx_buff, cx_xtra_doc )

* ... data set information for all variables
	dset = cx_data_set( cx_buff )
	IF (  dset .NE. unspecified_int4
     .	.AND. dset .NE. pdset_irrelevant ) THEN
* ... name of data set
	   CALL GET_DSET_NAME( dset, line_buff, alen )
	   CALL SPLIT_LIST(pttmode_explct, list_lun,
     .		'             DATA SET: '//line_buff(:alen), alen+23)
 3020	   FORMAT( 13X,A)
* ... title of data set
	   CALL GET_DSET_TITLE( dset, line_buff, alen )
	   IF ( alen .GT. 1 ) CALL SPLIT_LIST(pttmode_explct, list_lun,
     .		'             '//line_buff(:alen), alen+13)
	ENDIF

* ... location information in common for all variables
	DO 70 i = 1, nferdims
	   idim = perm(i)
	   cx = ABS( valid_ax(i) )
           IF ( cx .LE. 0 ) GOTO 70     !  6/92 bug fix
	   IF ( cx_lo_ww(idim,cx_buff).EQ.unspecified_val8
     .	   .OR. cx_lo_ww(idim,cx_buff).EQ.unspec_val8_2 ) GOTO 70
	   CALL ASCII_LIMITS( cx_buff, idim, line_buff, vlen )
	   col_head_format = AX_TITLE( idim, cx_grid(cx), alen )
	   CALL SPLIT_LIST(pttmode_explct, list_lun,
     .		'             '//col_head_format( :alen )//
     .				': '// line_buff(1:vlen), 0)
 70	CONTINUE

* ... then variable titles and any special limits or data set for a variable
	DO 60 ivar = 1, num_uvars_in_cmnd
	   cx = cx_list(ivar)
	   line_buff = vcode(ivar)
	   alen = TM_LENSTR1( vcode(ivar) )
	   IF ( INDEX(line_buff(:alen),'#') .EQ. 0 ) line_buff =
     .				KEY_STRING(cx, cx_xtra_doc, alen )
	   vname_buff   = VAR_TITLE( cx )
	   vlen   = TM_LENSTR1( vname_buff )
	   vname_buff   = vname_buff(:vlen)//' '//
     .			PAREN_SURROUND(VAR_UNITS(cx), slen1)
	   vlen = vlen+1+slen1
	   vname_buff   =  vname_buff(:vlen)//
     .			PAREN_SURROUND(VAR_TITLE_MOD(cx),slen2)
	   vlen = vlen+slen2
	   risc_buff = ' '

	   bad_buff = ' '
	   blen = 1
! do not write a bad-flag notation for string variables.
           IF ((do_comma_del .OR. do_tab_del) .AND. 
     .         (mr_type(mr_list(ivar)) .NE. ptype_string)) THEN
             bad_buff = '   BAD FLAG : '// 
     .                  TM_FMT(bad(ivar),prec_digits,16,blen)
             blen = blen + 14
           ENDIF

	   IF ( line_buff(:vlen) .EQ. vname_buff(:vlen) ) THEN
	      WRITE ( risc_buff, 3028 ) ivar, line_buff(:alen), 
     .           bad_buff(:blen)
	      CALL SPLIT_LIST(pttmode_explct, list_lun, risc_buff, 0)
 3028         FORMAT (' Column',I3,': ',A, A) 
	   ELSE	      
	      WRITE ( risc_buff, 3030 ) ivar,
     .           line_buff(:alen), vname_buff(:vlen),bad_buff(:blen)
	      CALL SPLIT_LIST(pttmode_explct, list_lun, risc_buff, 0)
 3030	      FORMAT (' Column',I3,': ',A,' is ',A, A)
	   ENDIF
 60	CONTINUE


 
* * * * * LIST DATA UNFORMATTED
 200    IF ( do_unform ) THEN
           point = 0
           DO i6 = lo6, hi6, del6
           DO i5 = lo5, hi5, del5
           DO i4 = lo4, hi4, del4
           DO i3 = lo3, hi3, del3
           DO i2 = lo2, hi2, del2
           DO i1 = lo1, hi1, del1
              IF ( interrupted ) RETURN
* default memory pointer for full-region variables
              point = point + 1

* loop through all variables for each record
              DO 250 ivar = 1, num_uvars_in_cmnd
                 IF (mr_type(mr_list(ivar)) .EQ. ptype_string) GOTO 5040
* ... optimized calculation for full-region variables
                 IF ( xact_range(ivar) ) THEN
                    out(ivar) = memry(mr_list(ivar))%ptr(point)
                 ELSE
* ... separate pointer calcs for non-full region variables
* ... e.g. a 2D field listed with a 3D will replicate at each 3rd-D location
                    iwd = 0
                    IF ( has_size(nferdims,ivar) ) iwd = (pt(nferdims)-lo(nferdims))
                    DO 240 i = nferdims-1, 1, -1
                    iwd = iwd*size(i,ivar)
                    IF ( has_size(i,ivar) ) iwd = iwd + (pt(i)-lo(i))
 240                CONTINUE
                    out(ivar) = memry(mr_list(ivar))%ptr(iwd+1)
                 ENDIF
 250          CONTINUE
              IF ( do_unform ) THEN
                 WRITE (list_lun, ERR=5010)
     .                  ( out(ivar),ivar = 1, num_uvars_in_cmnd )
              ELSE

                 WRITE (list_lun, list_format, ERR=5010)
     .                  ( out(ivar),ivar = 1, num_uvars_in_cmnd )
              ENDIF
           ENDDO
           ENDDO
           ENDDO
           ENDDO
           ENDDO
           ENDDO
* done - go home
           RETURN
        ENDIF


* * * * * LIST DATA WITH USER FORMAT (no frills)
 201	CONTINUE

	IF ( list_format_given ) THEN

	   risc_buff = ' '  ! this will store the value of any string outputs
	   line_buff = ' '  ! this will store a list of the internal variables to write, using format
	      
	   point = -1

	   DO i6 = lo6, hi6, del6
	   DO i5 = lo5, hi5, del5
	   DO i4 = lo4, hi4, del4
	   DO i3 = lo3, hi3, del3
	   DO i2 = lo2, hi2, del2
	   DO i1 = lo1, hi1, del1
	      IF ( interrupted ) RETURN
* default memory pointer for full-region variables
	      point = point + 1
	      lloc = 1

* loop through all variables for each record
	      DO 251 ivar = 1, num_uvars_in_cmnd

	      IF ( mr_type(mr_list(ivar)) .EQ. ptype_string ) THEN
* ... string variable
	 
		 IF ( xact_range(ivar) ) THEN
	            iwd = point
                 ELSE	
	            iwd = 0
	            IF ( has_size(nferdims,ivar) ) iwd = (pt(nferdims)-lo(nferdims))
	            DO i = nferdims-1, 1, -1
                       iwd = iwd*size(i,ivar)
                       IF ( has_size(i,ivar) ) iwd = iwd + (pt(i)-lo(i))
	            ENDDO
	         ENDIF

	         slen1 = GET_OFFSET_C_STRING_LEN(mr_c_pointer(mr_list(ivar)),
     .		  			 iwd )
	         risc_buff = ' '
		 CALL GET_OFFSET_C_STRING(mr_c_pointer(mr_list(ivar)),
     .				   iwd,
     .				   risc_buff(:slen1),
     .				   slen1     )

		 fmtstring = '(A)'
		 IF (num_it.EQ.num_uvars_in_cmnd ) fmtstring = 
     .            '( ' // list_format(item_start(ivar):item_end(ivar)) //  ')'

* The format may cut the string off shorter than its full length. Use the format length.
		 WRITE (buff2, fmtstring) risc_buff(:slen1)
		 slen1 = TM_LENSTR1(buff2) 
		 WRITE (line_buff(lloc:), fmtstring) buff2(:slen1)
		 lloc = lloc+slen1

	      ELSE

* ... numerical variable
* ... optimized calculation for full-region variables
	        IF ( xact_range(ivar) ) THEN
	           out(ivar) = memry(mr_list(ivar))%ptr(point+1)
	        ELSE
* ... separate pointer calcs for non-full region variables
* ... e.g. a 2D field listed with a 3D will replicate at each 3rd-D location
	           iwd = 0
	           IF ( has_size(nferdims,ivar) ) iwd = (pt(nferdims)-lo(nferdims))
	           DO i = nferdims-1, 1, -1
	              iwd = iwd*size(i,ivar)
	              IF ( has_size(i,ivar) ) iwd = iwd + (pt(i)-lo(i))
	           ENDDO
	           out(ivar) = memry(mr_list(ivar))%ptr(iwd+1)
	        ENDIF
		
		IF (num_it.EQ.num_uvars_in_cmnd ) fmtstring = 
     .               '( ' // list_format(item_start(ivar):item_end(ivar)) //  ')'

		IF (has_string .OR. has_int) THEN
		    IF ( TM_HAS_STRING (fmtstring, 'I') ) THEN
		       ival = INT(out(ivar)) 
		       WRITE (risc_buff, fmtstring) ival
		    ELSE
		       WRITE (risc_buff, fmtstring) out(ivar)
		    ENDIF
		    slen1 = TM_LENSTR1(risc_buff) 
		    line_buff(lloc:) = risc_buff(:slen1)
		    lloc = lloc + slen1
	         ENDIF

	      ENDIF  ! string or numeric

 251	      CONTINUE

	      IF (.NOT. (has_string.OR.has_int)) THEN
                 WRITE (list_lun, list_format, ERR=5010)
     .                  ( out(ivar),ivar = 1, num_uvars_in_cmnd )
	      ELSE
	         lloc = TM_LENSTR1(line_buff)
                 buff2 = TM_FMT(DBLE(lloc),0,16,vlen)
	         fmtstring = '(A'//buff2(:vlen)//')'
	         WRITE (list_lun, fmtstring, ERR=5010) line_buff
	      ENDIF

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* done - go home
	   RETURN
	ENDIF

* * * * PREPARE FORMATS ADAPTED TO THE DATA AND AXIS RANGES
* ... coordinate labelling at left of each record
	col_ax = perm(1)
	cx = valid_ax(1)
	CALL ROW_COORD_FMT( col_ax, cx, lo(1), hi(1), 1,
     .			    row_ss_fmt, col_dec, row_ww_wid, cleft )
	data_len = 0
        IF (norow) cleft = 0

* ... column labelling - variable codes for headings - FORTRAN formats for data
	DO 400 ivar = 1, num_uvars_in_cmnd

* ... determine numeric format to fit nicely in each field
* ... first find largest value to be printed
	   IF ( mr_type(mr_list(ivar)) .EQ. ptype_string ) THEN
	       nleft = GET_MAX_STRING_LEN(cx_list(ivar),
     .					  mr_list(ivar),
     .					  memry(mr_list(ivar))%ptr )
	   ELSE 
	      CALL MINMAX( memry(mr_list(ivar))%ptr, msize(ivar),
     .			   bad(ivar), small, big, nok)
	      all_bad = nok .EQ. 0

* . . . how many digits right and left of decimal point ?
	      biggest = MAX( ABS( small ), big )
	      CALL GET_SIG_DIGITS( biggest, ABS(prec_digits), nleft, nright )
	      nleft = MAX( 1, nleft )			! at least "0" to left
              IF ( prec_digits .GT. 0 ) THEN
*                positive means user is giving the total precision
                 ndigits = prec_digits
              ELSE
*                 zero or negative means user is giving nright; use nleft from above 
*                 (independent of prec_digits) and compute the total number of digits
                  nright = ABS(prec_digits)
                  ndigits = nleft + nright
              ENDIF
	   ENDIF

* ... field must be wide enough to accomodate the variable code heading
*     (to a max of 8 characters)
	   min_width = MIN( 8, TM_LENSTR1( vcode(ivar) ) + 1 )

* . . . determine FORTRAN format for listing of this variable
	   IF ( mr_type(mr_list(ivar)) .EQ. ptype_string ) THEN
              width(ivar) = MIN(nleft+3, 1020)  ! arbitrary 1020 char max   
	   ELSEIF ( all_bad ) THEN
	      width(ivar) = MAX(min_width, 5)
	      WRITE ( bad_fmt(ivar), 3002 ) 5
 3002	FORMAT ( '(A',I2,')' )	
	   ELSE
	      CALL VAR_DATA_FMT( nleft, nright, ndigits, min_width,
     .			         width(ivar), good_fmt(ivar), bad_fmt(ivar) )
	   ENDIF

	   data_len = data_len + width(ivar)

 400	CONTINUE

* . . . will it fit ?
	IF ( data_len .GT. line_length-cleft ) GOTO 5030
	
* . . . write a heading line of variable codes
	IF ( heading ) THEN
           line_buff = ' '

           IF (do_comma_del .OR. do_tab_del) THEN
* ... Write entire variable names without cutting them off.

              slen1 = TM_LENSTR1( vcode(1) )
	      w0 = cleft - slen1 + MIN(width(1), slen1)
              IF (w0 .LE. 0 .OR. cleft .EQ. 0) w0 = 1
              DO 420 ivar = 1, num_uvars_in_cmnd

                 slen1 = TM_LENSTR1(vcode(ivar))
                 line_buff(w0:w0+slen1) = vcode(ivar)(:slen1)

                 IF (ivar .LT. num_uvars_in_cmnd)
     .              line_buff(w0+slen1:w0+slen1) = tab_or_comma
                 w0 = w0 + slen1 + 1
 420	      CONTINUE

           ELSE
* ... Not tab or comma delimited: use width of each field
 
	      w0 = cleft
	      slen1 = MIN( w0, TM_LENSTR1( vcode(1) ) )
              IF (cleft .EQ. 0) slen1 = TM_LENSTR1( vcode(1) )
	      w0 = w0 + width(1)
	      line_buff(w0-slen1:w0-1) = vcode(1)(:slen1)
              w0 = w0 + 1
              IF (num_uvars_in_cmnd .GT. 1) line_buff(w0:w0) = ' '

	      DO 421 ivar = 2, num_uvars_in_cmnd
                 line_buff(w0:w0) = ' '

                 slen2 = 0
                 IF ( TM_LENSTR1(vcode(ivar)) .LT. width(ivar) ) 
     .               slen2 = (width(ivar) - TM_LENSTR1(vcode(ivar)))/2
                 line_buff(w0+1+slen2:w0+width(ivar)) = vcode(ivar)

                 IF (ivar .EQ. num_uvars_in_cmnd)
     .              line_buff(w0+1+slen2:) = vcode(ivar)
                 w0 = w0 + width(ivar)
 421	      CONTINUE

           ENDIF


           w0 = TM_LENSTR1(line_buff) 
	   CALL SPLIT_LIST(pttmode_explct, list_lun, line_buff(1:w0), w0)
	ENDIF
        line_buff = ' '

* * * * *  WRITE FORMATTED DATA LINES * * * * *
* each line has format: "coord/sub: data1 data2 data3 ..."
* for a 1D field a single block of such lines is printed
* for a 2D field each block is separated by an identifier of the 3rd axis
* for a 3D field each block of blocks is further separated and identified
* etc. for a 4D field

* If the /NOROWLAB qualifier was given, do not do the "coord/sub:" on each data line

 3050   FORMAT (1X,4('-'),1X,A)
	point = -1
	DO i6 = lo6, hi6, del6
* label the block of block of blocks (6th dimension)
	   risc_buff = ' '
	   IF ( hi6 .GT. lo6 ) THEN
	      CALL POS_LAB( i6, valid_ax(6), perm(6),
     .			    ax_dec_pt(perm(6)), line_buff, alen )
	      WRITE (risc_buff, 3050) line_buff(:alen)
	      CALL SPLIT_LIST(pttmode_explct, list_lun, risc_buff, 0)
	   ENDIF
	DO i5 = lo5, hi5, del5
* label the block of block of blocks (5th dimension)
	   risc_buff = ' '
	   IF ( hi5 .GT. lo5 ) THEN
	      CALL POS_LAB( i5, valid_ax(5), perm(5),
     .			    ax_dec_pt(perm(5)), line_buff, alen )
	      WRITE (risc_buff, 3050) line_buff(:alen)
	      CALL SPLIT_LIST(pttmode_explct, list_lun, risc_buff, 0)
	   ENDIF
	DO i4 = lo4, hi4, del4
* label the block of block of blocks (4th dimension)
	   risc_buff = ' '
	   IF ( hi4 .GT. lo4 ) THEN
	      CALL POS_LAB( i4, valid_ax(4), perm(4),
     .			    ax_dec_pt(perm(4)), line_buff, alen )
	      WRITE (risc_buff, 3050) line_buff(:alen)
	      CALL SPLIT_LIST(pttmode_explct, list_lun, risc_buff, 0)
	   ENDIF
	DO i3 = lo3, hi3, del3
* label the block of blocks (3rd dimension)
	   IF ( hi3 .GT. lo3 ) THEN
	      CALL POS_LAB( i3, valid_ax(3), perm(3),
     .			    ax_dec_pt(perm(3)), line_buff, alen )
	      WRITE (risc_buff, 3050) line_buff(:alen)
	      CALL SPLIT_LIST(pttmode_explct, list_lun, risc_buff, 0)
	   ENDIF
	DO i2 = lo2, hi2, del2
* label the block of blocks (3rd dimension)
	   IF ( hi2 .GT. lo2 ) THEN
	      CALL POS_LAB( i2, valid_ax(2), perm(2),
     .			    ax_dec_pt(perm(2)), line_buff, alen )
	      WRITE (risc_buff, 3050) line_buff(:alen)
	      CALL SPLIT_LIST(pttmode_explct, list_lun, risc_buff, 0)
	   ENDIF
	DO i1 = lo1, hi1, del1
           line_buff = ' '
	   IF ( interrupted ) RETURN

*  create carefully sized label for this row
	   cx = valid_ax(1)

           IF (.NOT. norow) THEN
	      IF ( cx .GT. 0 ) THEN
	         CALL TRANSLATE_TO_WORLD( 
     .			TM_WORLD( i1, cx_grid(cx), perm(1), box_middle ),
     .			perm(1), cx_grid(cx), col_dec, line_buff(:row_ww_wid) )
	         WRITE ( line_buff(row_ww_wid+1:cleft), row_ss_fmt ) i1
	      ELSE
	         line_buff(1:1) = ss_dim_name(perm(1))
	         IF ( i1 .NE. unspecified_int4 ) THEN
	            WRITE ( line_buff(2:cleft), row_ss_fmt ) i1	! e.g. "I / 15:"
	         ELSE
	            line_buff(2:) = ' / *:'			! e.g. "I / *:"
	         ENDIF 
	      ENDIF
           ENDIF
           w0 = cleft

* default memory pointer for full-region variables
	   point = point + 1

* loop through all variables for each record
           ww = w0 + 1
	   DO 550 ivar = 1, num_uvars_in_cmnd
	      w = w0 + width(ivar) ! points to last column of field
	      IF ( mr_type(mr_list(ivar)) .EQ. ptype_string ) THEN
* ... string variable
	         IF ( xact_range(ivar) ) THEN
	            iwd = point
                 ELSE	
	            iwd = 0
	            IF ( has_size(nferdims,ivar) ) iwd = (pt(nferdims)-lo(nferdims))
	            DO 530 i = nferdims-1, 1, -1
                       iwd = iwd*size(i,ivar)
                       IF ( has_size(i,ivar) ) iwd = iwd + (pt(i)-lo(i))
 530	            CONTINUE
	         ENDIF

                 IF (do_comma_del .OR. do_tab_del) THEN
* ... tab or comma formatting
                    line_buff(ww:ww) = '"'
	            slen1 = GET_OFFSET_C_STRING_LEN(mr_c_pointer(mr_list(ivar)),
     .					 iwd )
	            CALL GET_OFFSET_C_STRING(mr_c_pointer(mr_list(ivar)),
     .				   iwd,
     .				   line_buff(ww+1:),
     .				   slen1    )
                    line_buff(ww+slen1+1:ww+slen1+1) = '"'

                    IF (ivar .LT. num_uvars_in_cmnd) 
     .                 line_buff(ww+slen1+2:ww+slen1+2) = tab_or_comma
                    ww = ww + slen1 + 3
                    w = ww               
                 ELSE
* ... not tab or comma formatting

	            line_buff(w0+2:w0+2) = '"'
	            line_buff(w0+3:w) = ' '
	            slen1 = GET_OFFSET_C_STRING_LEN(mr_c_pointer(mr_list(ivar)),
     .		  			 iwd )
	            CALL GET_OFFSET_C_STRING(mr_c_pointer(mr_list(ivar)),
     .				   iwd,
     .				   line_buff(w0+3:),
     .				   width(ivar)-2     )
	            IF (slen1 .GT. width(ivar)-2) THEN
	               line_buff(w-3:w) = '..."'
	            ELSE
	               line_buff(w0+slen1+3:w0+slen1+3) = '"'
	            ENDIF
                 ENDIF  ! tab-or-comma
	      ELSE
* ... numerical variable
* ... optimized calculation for full-region variables
	        IF ( xact_range(ivar) ) THEN
	           out(ivar) = memry(mr_list(ivar))%ptr(point+1)
	        ELSE
* ... separate pointer calcs for non-full region variables
* ... e.g. a 2D field listed with a 3D will replicate at each 3rd-D location
	           iwd = 0
	           IF ( has_size(nferdims,ivar) ) iwd = (pt(nferdims)-lo(nferdims))
	           DO 540 i = nferdims-1, 1, -1
	           iwd = iwd*size(i,ivar)
	           IF ( has_size(i,ivar) ) iwd = iwd + (pt(i)-lo(i))
 540	           CONTINUE
	           out(ivar) = memry(mr_list(ivar))%ptr(iwd+1)
	        ENDIF
	        IF ( out(ivar) .EQ. bad(ivar) ) THEN
                   IF (do_comma_del .OR. do_tab_del) THEN
                       line_buff = line_buff(:ww-1)//
     .	                   TM_FMT(out(ivar),ndigits,48,vlen)
                       IF (ivar .LT. num_uvars_in_cmnd) 
     .                   line_buff(ww+vlen:ww+vlen) = tab_or_COMMA
     	               ww = ww+vlen+1
                       w = ww
                   ELSE
	              WRITE(line_buff(w0+1:w),bad_fmt(ivar)) '....'
                   ENDIF
	        ELSE
                   IF (do_comma_del .OR. do_tab_del) THEN
* ... tab or comma formatting
                       line_buff = line_buff(:ww-1)//
     .	                   TM_FMT(out(ivar),ndigits,48,vlen)
                       IF (ivar .LT. num_uvars_in_cmnd) 
     .                   line_buff(ww+vlen:ww+vlen) = tab_or_COMMA
     	               ww = ww+vlen+1
                       w = ww
                    ELSE
	              WRITE(line_buff(w0+1:w),good_fmt(ivar)) out(ivar)
                    ENDIF
	        ENDIF
	      ENDIF
	      w0 = w

 550	   CONTINUE

* write the fully formatted line
           IF (do_comma_del .OR. do_tab_del) w0 = ww
	   CALL SPLIT_LIST(pttmode_explct, list_lun, line_buff, w0)
           line_buff = ' '

           ENDDO
           ENDDO
           ENDDO
           ENDDO
           ENDDO
           ENDDO

* successful completion
	RETURN

* error exits
 5000	RETURN
 5010	CALL ERRMSG( ferr_erreq, status, list_format, *5000 )
 5020	ss_save = cx_by_ss(idim,cx)
	cx_by_ss(idim,cx) = .TRUE.
	buff1 = CX_DIM_STR( idim,cx,':',no_decims,slen1 )
	cx_by_ss(idim,cx) = ss_save
	cx_by_ss(idim,cx_buff) = .TRUE.
	cx_lo_ss(cx_buff,idim) = lo(i)
	cx_hi_ss(cx_buff,idim) = hi(i)
	buff2 = CX_DIM_STR( idim,cx_buff,':',no_decims,slen2 )
	buff3 = LEFINT(ivar,slen3)
	CALL ERRMSG( ferr_inconsist_plane, status,
     .		ww_dim_name(idim)//' axis'//pCR//
     .		'Variable '//buff3(:slen3)//' is '//buff1(:slen1)//pCR//
     .		'expression is '//buff2(:slen2), *5000 )
 5030	CALL ERRMSG( ferr_invalid_command, status,
     .		'list of variables too long for output lines'//pCR//
     .		'Reduce SET LIST/PRECISION or specify /FORMAT', *5000 )
 5040	CALL ERRMSG( ferr_invalid_command, status,
     .		'unformatted output is for numeric data only',
     .		*5000 )
 5050	CALL ERRMSG( ferr_invalid_command, status,
     . 'Fortran format for list using Integer type'//pCR//
     .		'must have one specifier per variable',
     . *5000 )

	END
