	SUBROUTINE LIST_MULTI_SUB(memory, mr_list,cx_list,heading,
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

	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret.parm'
	include 'errmsg.parm'
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
	PARAMETER     ( line_length  = 500,
     .			no_decims = .FALSE. )
	INTEGER		slash_precision, slash_norow
	PARAMETER     (	slash_precision = 17+1,
     .                  slash_norow   = 32+1 )

* calling argument declarations:
	LOGICAL	heading
	INTEGER	mr_list(num_uvars_in_cmnd), cx_list(num_uvars_in_cmnd),
     .		perm(4), ws, status
	REAL	memory( mem_blk_size, max_mem_blks )
* ... work arrays ...
	LOGICAL		xact_range(ws), has_size(4,ws)
	INTEGER		pt_blk(ws), size(4,ws), msize(ws), width(ws)
	REAL		out(ws), bad(ws)
* ss_save removed from integer declaration and put in logical decl.  *kob* 8/94	
* V500 *kob* 3/99- up VAR_CODE and vcode decl's to 64 chars and KEY_STRING
*                  to 128 and VAR_UNITS to 64
* V552 *acm* 4/03- up VAR_CODE and vcode decl's from 64 to 128 chars
* local variable declarations:
	LOGICAL		all_bad, do_unform, ss_save, do_tab_del,
     .                  do_comma_del, norow
	INTEGER		TM_LENSTR1, CGRID_AXIS, GET_MAX_STRING_LEN,
     .			GET_OFFSET_C_STRING_LEN, 
     .			w0, w, data_len, nleft, nright, alen, blen,
     .			lo(4), hi(4), del(4), pt(4), valid_ax(4),
     .			idim, vlen, cleft, col_ax, col_dec,
     .			cx, slen1, slen2, slen3, i, 
     .			ivar, point, word, block, iwd, iblk, dset, ax, new_ax,
     .			min_width, row_ww_wid, lp, prec_digits, ww
	CHARACTER*255	VAR_TITLE, VAR_TITLE_MOD, PAREN_SURROUND, vname_buff
	CHARACTER	KEY_STRING*128, AX_TITLE*32, VAR_UNITS*64, LEFINT*8,
     .			CX_DIM_STR*16, VAR_CODE*128, TM_FMT*16,
     .			buff1*16, buff2*16, buff3*16, line_buff*1024,
     .			row_ss_fmt*14, bad_ax*5, bad_buff*32,
     .			col_head_format*32, sep*1
	REAL		big, small, biggest, value
	REAL*8		TM_WORLD
	CHARACTER	good_fmt(max_intrp)*10, bad_fmt(max_intrp)*10,
     .			vcode(max_intrp)*128, tab_or_comma*1
#ifdef DEC_FORTRAN
	INTEGER		ios
#endif
* convenience equivalence:
	INTEGER		lo1, lo2, lo3, lo4, hi1, hi2, hi3, hi4,
     .			del1, del2, del3, del4,
     .			i1, i2, i3, i4
	EQUIVALENCE	(i1, pt(1)) , (i2, pt(2)) , (i3, pt(3)),  (i4,pt(4)),
     .			(lo1, lo(1)), (lo2, lo(2)), (lo3,lo(3)),  (lo4,lo(4)),
     .			(hi1, hi(1)), (hi2, hi(2)), (hi3,hi(3)),  (hi4,hi(4)),
     .			(del1,del(1)),(del2,del(2)),(del3,del(3)),(del4,del(4))

* initialize
	status		= ferr_ok
        do_unform       = list_fmt_type .EQ. plist_unformatted
	risc_buff = ' '

        do_tab_del   = list_fmt_type .EQ. plist_tab_del
        do_comma_del = list_fmt_type .EQ. plist_comma_del

        norow = qual_given(slash_norow) .GT. 0

        IF ( do_comma_del .OR. do_tab_del ) THEN
* ... column headings for tab and comma-delimited output (9/95)
	   IF ( do_comma_del ) THEN
	      tab_or_comma = ','
	   ELSEIF( do_tab_del ) THEN
	      tab_or_comma = CHAR(9)
           ENDIF
              
	ENDIF

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

* determine list limits checking for conformable shapes of data to list
* ... for each axis each var must be same size or length 1
	DO 10 i = 1, 4
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
	DO 20 i = 1, 4
	   idim = perm(i)
	   DO 18 ivar = 1, num_uvars_in_cmnd
	      cx = cx_list(ivar)
	      has_size( i, ivar ) = cx_hi_ss(cx,idim) .GT. cx_lo_ss(cx,idim)
	      size( i, ivar ) = cx_hi_ss(cx,idim) - cx_lo_ss(cx,idim) + 1
 18	   CONTINUE
 20	CONTINUE
* ... and total size
	DO 25 ivar = 1, num_uvars_in_cmnd
	msize(ivar) = 1
	DO 25 idim = 1, 4
 25	msize(ivar) = msize(ivar) * size(idim,ivar)	

* determine which axes have unambiguous coordinates (all variables that are
* not point locations on this axis must have the same coordinates)
* create a string (bad_ax) with a list of ambiguous axes and a list, valid_ax,
* of context pointers for axes with valid lists of coordinates
	sep = ' '
	bad_ax = ' '
	slen1 = 1	! length of bad axis string
	DO 30 i = 1, 4
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
	   DO 38 i = 1, 4
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
	   pt_blk(ivar) = mr_blk1( mr_list(ivar) )	
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
	DO 70 i = 1, 4
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
           IF (do_comma_del .OR. do_tab_del) THEN
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

* * * * * LIST DATA UNFORMATTED OR WITH USER FORMAT (no frills)
* Note: cannot direct user format to GUI because records may be
* arbitrarily long
 200	IF ( do_unform .OR. list_format_given ) THEN
	   point = -1                    ! bug fix 4/93
	   DO 300 i4 = lo4, hi4, del4
	   DO 300 i3 = lo3, hi3, del3
	   DO 300 i2 = lo2, hi2, del2
	   DO 300 i1 = lo1, hi1, del1
	      IF ( interrupted ) RETURN
* default memory pointer for full-region variables
	      point = point + 1
	      block = point / mem_blk_size
	      word  = point - block*mem_blk_size

* loop through all variables for each record
	      DO 250 ivar = 1, num_uvars_in_cmnd
	         IF (mr_type(mr_list(ivar)) .EQ. ptype_string) GOTO 5040
* ... optimized calculation for full-region variables
	         IF ( xact_range(ivar) ) THEN
	            out(ivar) = memory( word+1, pt_blk(ivar)+block ) 
	         ELSE
* ... separate pointer calcs for non-full region variables
* ... e.g. a 2D field listed with a 3D will replicate at each 3rd-D location
	            iwd = 0
	            IF ( has_size(4,ivar) ) iwd = (pt(4)-lo(4))
	            DO 240 i = 3, 1, -1
	            iwd = iwd*size(i,ivar)
		    IF ( has_size(i,ivar) ) iwd = iwd + (pt(i)-lo(i))
 240	            CONTINUE
	            iblk = iwd / mem_blk_size
	            iwd  = iwd - iblk*mem_blk_size
	            out(ivar) = memory( iwd+1, pt_blk(ivar)+iblk ) 
	         ENDIF
 250	      CONTINUE
	      IF ( do_unform ) THEN
	         WRITE (list_lun, ERR=5010)
     .			( out(ivar),ivar = 1, num_uvars_in_cmnd )
	      ELSE
#ifdef DEC_FORTRAN
* on DEC compilers a field too small for the value generates an ERR= branch
	         WRITE (list_lun, list_format, IOSTAT=ios)
     .			( out(ivar),ivar = 1, num_uvars_in_cmnd )
	        IF ( ios.NE.0 .AND. ios.NE.63 ) GOTO 5010  ! 63 = fmt conv err 
#else
	         WRITE (list_lun, list_format, ERR=5010)
     .			( out(ivar),ivar = 1, num_uvars_in_cmnd )
#endif
	      ENDIF
 300	   CONTINUE
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
     .					  memory(1,pt_blk(ivar)) )
	   ELSE 
	      CALL MINMAX( memory(1,pt_blk(ivar)), msize(ivar),
     .			   bad(ivar), small, big )
	      all_bad = small .EQ. arbitrary_large_val4

* . . . how many digits right and left of decimal point ?
	      biggest = MAX( ABS( small ), big )
	      CALL GET_SIG_DIGITS( biggest, prec_digits, nleft, nright )
	      nleft = MAX( 1, nleft )			! at least "0" to left
	   ENDIF

* ... field must be wide enough to accomodate the variable code heading
*     (to a max of 8 characters)
	   min_width = MIN( 8, TM_LENSTR1( vcode(ivar) ) + 1 )

* . . . determine FORTRAN format for listing of this variable
	   IF ( mr_type(mr_list(ivar)) .EQ. ptype_string ) THEN
              width(ivar) = MIN(nleft+3, 1020)  ! arbitrary 1020 char max   
	   ELSEIF ( all_bad ) THEN
	      width(ivar) = min_width
	      WRITE ( bad_fmt(ivar), 3002 ) min_width
 3002	FORMAT ( '(A',I2,')' )	
	   ELSE
	      CALL VAR_DATA_FMT( nleft, nright, prec_digits, min_width,
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

	point = -1                    ! bug fix 4/93
	DO 600 i4 = lo4, hi4, del4
* label the block of block of blocks (4th dimension)
	   risc_buff = ' '
	   IF ( hi4 .GT. lo4 ) THEN
	      CALL POS_LAB( i4, valid_ax(4), perm(4),
     .			    ax_dec_pt(perm(4)), line_buff, alen )
	      WRITE (risc_buff, 3050) line_buff(:alen)
 3050	      FORMAT (1X,4('-'),1X,A)
	      CALL SPLIT_LIST(pttmode_explct, list_lun, risc_buff, 0)
	   ENDIF
	DO 600 i3 = lo3, hi3, del3
* label the block of blocks (3rd dimension)
	   IF ( hi3 .GT. lo3 ) THEN
	      CALL POS_LAB( i3, valid_ax(3), perm(3),
     .			    ax_dec_pt(perm(3)), line_buff, alen )
	      WRITE (risc_buff, 3050) line_buff(:alen)
	      CALL SPLIT_LIST(pttmode_explct, list_lun, risc_buff, 0)
	   ENDIF
	DO 600 i2 = lo2, hi2, del2
* label the block of blocks (3rd dimension)
	   IF ( hi2 .GT. lo2 ) THEN
	      CALL POS_LAB( i2, valid_ax(2), perm(2),
     .			    ax_dec_pt(perm(2)), line_buff, alen )
	      WRITE (risc_buff, 3050) line_buff(:alen)
	      CALL SPLIT_LIST(pttmode_explct, list_lun, risc_buff, 0)
	   ENDIF
	DO 600 i1 = lo1, hi1, del1
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
	   block = point / mem_blk_size
	   word  = point - block*mem_blk_size

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
	            IF ( has_size(4,ivar) ) iwd = (pt(4)-lo(4))
	            DO 530 i = 3, 1, -1
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
	           out(ivar) = memory( word+1, pt_blk(ivar)+block ) 
	        ELSE
* ... separate pointer calcs for non-full region variables
* ... e.g. a 2D field listed with a 3D will replicate at each 3rd-D location
	           iwd = 0
	           IF ( has_size(4,ivar) ) iwd = (pt(4)-lo(4))
	           DO 540 i = 3, 1, -1
	           iwd = iwd*size(i,ivar)
	           IF ( has_size(i,ivar) ) iwd = iwd + (pt(i)-lo(i))
 540	           CONTINUE
	           iblk = iwd / mem_blk_size
	           iwd  = iwd - iblk*mem_blk_size
	           out(ivar) = memory( iwd+1, pt_blk(ivar)+iblk ) 
	        ENDIF
	        IF ( out(ivar) .EQ. bad(ivar) ) THEN
                   IF (do_comma_del .OR. do_tab_del) THEN
                       line_buff = line_buff(:ww-1)//
     .	                   TM_FMT(out(ivar),prec_digits,16,vlen)
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
     .	                   TM_FMT(out(ivar),prec_digits,16,vlen)
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
 600	   CONTINUE

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
     .		'format disallowed for multiple variables w/ strings',
     .		*5000 )
	END
