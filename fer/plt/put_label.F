	SUBROUTINE PUT_LABEL(	labnum,
     .				xpos, ypos,
     .				justify,
     .				labht,
     .				string,
     .				xaxrel, yaxrel )

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
* define a single label to PPLUS after checking to see if it needs to be shrunk
* save its definition in a PPLUS symbol for customizing by user
* possibly place the label on the plot relative to axis lengths

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/14/87
* revision 0.1 - 9/23/87 - removed @AS - implied via DFLTFNT command
* revision 0.2 - 3/23/88 - replaced @AS using SET MODE FONT_ASCII
* revision 0.3 - 4/7/88  - added PPLUS symbol definition
* revision 0.4 - 9/8/88  - added *axrel - relative to axis length calc.
* revision 0.5 - 9/21/88 - no action if no_labels
* V200:   1/3/89 - label numbers maintained internally
*		 - no automatic shrinking of text string (maxlen removed)
*		 - plot5_cmnd --> ppl_buff
*		 - AS and AC fonts put into label but not into symbol
* Unix/RISC port - 2/91 - cannot "//" with CHAR*(*) variables
*                       - cannot use char functions in write stmts
* V320: 8/18/94	- *kob* IBM port - need to use intrinsic function REAL instead
*			           of SNGL for ibm
* V400: 6/22/95 - *sh* define movable label numbers as symbols when requested
*		  (requested by subtracting axis number from ppl_movlab)
* V420: 2/1/96  - *sh* If "string" is blank output "<ESC>@AC" to wipe out
*		   previous label
* V510: *sh8 3/00 - add DODS URL label
* V533: *sh8 7/01 - if label contains single tic marks double them so PLOT+
*		    doesn't try to interpret symbols
* 2/03 *kob* - g77 port - g77 won't allow intrinsic functions in PARAMETER
*                         statements.  use an octal constant instead 
* 7/10 *acm* - ticket 1747: LABNUM_CALEN for automatically generated CALENDAR
*              label, points to the PPL LAB# number.
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus.
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
*      *kms* 5/12 Always use generic REAL instead of specific SNGL 
*                 (SNGL must have a double-precision or complex argument)
* V68  *acm* 5/12  Dont write the font to xlab, ylab if they are blank strings.
* V686 *acm* 12 13 set up the label symbols LABNUM_E and LABNUM_F
* V6.95 *acm* 3/15 Changes for PyFerret fonts. 
* V7.45 *acm*  3/19 Fix issue# 1915: When pyferret fonts, send moveable label 
*                   strings starting with @PY. Then labels starting with numeric 
*                   characters don't lose those numbers .

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'plot_setup.parm'
	include 'xprog_state.cmn'
	include 'xplot_setup.cmn'
	include 'xtext_info.cmn'
	include 'axis_inc.decl'	! axis lengths
	include 'AXIS.INC'		! axis lengths
	include 'xrisc.cmn'		! to fix RISC FORTRAN bug
        include 'pyfonts.cmn'

* PLOT+ parameter fom PPLV11INC:PARAMPL5.DAT
	INTEGER		nlabs
	PARAMETER     ( nlabs=50 )

* calling argument declarations:
	INTEGER		labnum, justify, xaxrel, yaxrel
	REAL		xpos, ypos, labht
	CHARACTER*(*)	string

* internal variable declarations:
	LOGICAL		TM_DIGIT_TEST
	INTEGER		flen, lsym, ii
	CHARACTER	TM_FMT*48, font*3, buf4*4,
     .			labname(ppl_dset:ppl_calendar)*6
	REAL 		xp, yp

* local parameter declarations:
	INTEGER		maxsym, len_in, i
	CHARACTER*1	esc
	PARAMETER     ( esc = CHAR(27),
     .			maxsym = 102   )! max. symbol length for PPLUS (PPL bug)

* data initialization
	DATA labname(ppl_dset)/'DSET'/,
     .	     labname(ppl_aspect)/'ASPECT'/,
     .	     labname(ppl_year)/'YEAR'/,
     .	     labname(ppl_datitl)/'DATITL'/,
     .	     labname(ppl_offset)/'OFFSET'/,
     .	     labname(ppl_dodsurl)/'DODS'/,
     .	     labname(ppl_calendar)/'CALEND'/

* if label has single tic marks double them so PLOT+ doesnt interpret symbols
	IF ( INDEX(string,"'date'") .GT. 0 ) THEN
* ... hack to let PLOT+ date time symbols be evaluated (using single tics)
	  risc_buff = string
	  len_in = MIN( size_rbuff, LEN(string) )
	ELSE
	  risc_buff = ' '
	  len_in = MIN( size_rbuff-1, LEN(string) )
	  len_rbuff = 0
	  DO i = 1, len_in
	    len_rbuff = len_rbuff+1
	    risc_buff(len_rbuff:len_rbuff) = string(i:i)
	    IF (string(i:i) .EQ. "'") THEN
	      len_rbuff = len_rbuff+1
	      risc_buff(len_rbuff:len_rbuff) = "'" ! 2 tics together
	    ENDIF
	  ENDDO
	ENDIF

* adjust positions relative to axis lengths if requested
	xp = xpos
	yp = ypos
	IF ( xaxrel .EQ. lab_loc_axlen_plus ) THEN
	   xp = xp + xlen
	ELSEIF ( xaxrel .EQ. lab_loc_axlen_times ) THEN
	   xp = xp * xlen
	ENDIF
	IF ( yaxrel .EQ. lab_loc_axlen_plus ) THEN
	   yp = yp + ylen
	ELSEIF ( yaxrel .EQ. lab_loc_axlen_times ) THEN
	   yp = yp * ylen
	ENDIF

* PPLUST symbols strings cannot exceed maxsym
	lsym = MIN( maxsym, len_rbuff )

* select PLOT+ font
	IF ( string .EQ. ' ' ) THEN
	   font = esc // '@AC '
	ELSEIF ( .NOT.mode_font_ascii ) THEN
	   font = ' '
	ELSEIF ( labnum .EQ. ppl_title ) THEN
	   font = '@AC'
	ELSE
	   font = '@AS'
	ENDIF

	IF (pyfont) font = '@PY'

* Let user select the font if they want

        IF (string(1:1) .EQ. "@") THEN
           font = ' '
        ENDIF

* define the label to PPLUS
	IF     ( labnum .EQ. ppl_title ) THEN
* ... TITLE
	   WRITE ( ppl_buff, 3200) labht, font//risc_buff(:len_rbuff)
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	   WRITE ( ppl_buff, 3500 ) 'TIT', risc_buff(:lsym)
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	ELSEIF ( labnum .EQ. ppl_xlab ) THEN
* ... XLAB
	   IF (risc_buff(:len_rbuff) .NE. ' ') 
     .			CALL PPLCMD( from, line, 0,
     .			'XLAB '//font//risc_buff(:len_rbuff), 1, 1 )
	   WRITE ( ppl_buff, 3300 ) labht
	   CALL PPLCMD( from, line, 0, ppl_buff, 1, 1 )
	   WRITE ( ppl_buff, 3500 ) 'X', risc_buff(:lsym)
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	ELSEIF ( labnum .EQ. ppl_ylab ) THEN
* ... YLAB
	   IF (risc_buff(:len_rbuff) .NE. ' ') 
     .			CALL PPLCMD( from, line, 0,
     .			'YLAB '//font//risc_buff(:len_rbuff), 1, 1 )
	   WRITE ( ppl_buff, 3400 ) labht
	   CALL PPLCMD( from, line, 0, ppl_buff, 1, 1 )
	   WRITE ( ppl_buff, 3500 ) 'Y', risc_buff(:lsym)
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	ELSE
* ... LABS (PLOT+ movable labels)
	   nlabs_on = nlabs_on + 1
* ... check for PLOT+ limit (no error chack there)
	   IF ( nlabs_on .GT. nlabs ) CALL WARN( 'PLOT+ label limit exceeded' )
	   WRITE ( ppl_buff, 3000 ) nlabs_on,xp,yp,justify,
     .				    font//risc_buff(:len_rbuff)
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	   WRITE ( ppl_buff, 3100 ) nlabs_on, labht
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
           buf4 = TM_FMT(DBLE(nlabs_on),0,3,flen)      ! for Unix/RISC
	   WRITE ( ppl_buff, 3500 ) buf4(:flen),  risc_buff(:lsym)
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
* ... define a PLOT+ symbol containing the label number "SET LABNUM_X 4"
*	or LABNUM_DSET, LABNUM_YEAR, etc.
	   ii = ppl_movlab - labnum
	   IF ( ii.GE.1 .AND. ii.LE.nferdims ) THEN
	      WRITE ( ppl_buff, 3500 )
     .			"num_"//ww_dim_name(ii), buf4(:flen)
	      CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	   ELSEIF ( ii.GE.ppl_dset  .AND. ii.LE.ppl_calendar ) THEN
	      WRITE ( ppl_buff, 3500 )
     .			"num_"//labname(ii), buf4(:flen)
	      CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	   ENDIF
	ENDIF

* successful completion every time
	RETURN
 3000	FORMAT('LABS/NOUSER ',I3,1X,2(F6.2,1X),I3,1X,A)
 3100	FORMAT('HLABS ',I3,1X,F6.3 )
 3200	FORMAT('TITLE ',F6.3,1X,A)
 3300   FORMAT('LABSET,,', F6.3 )	! "LABSET,,x.x"
 3400   FORMAT('LABSET,,,', F6.3 )	! "LABSET,,,x.x"
 3500   FORMAT('SET LAB',A,1X, A)	! "SET LABxxx string"
	END
