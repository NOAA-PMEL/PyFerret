	SUBROUTINE PUT_LABEL(	labnum,
     .				xpos, ypos,
     .				justify,
     .				labht,
     .				string,
     .				xaxrel, yaxrel )

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

#ifdef unix 
	include 'ferret_cmn/plot_setup.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xplot_setup.cmn'
	include 'ferret_cmn/xtext_info.cmn'
	include 'tmap_pplv11inc/axis_inc.decl'	! axis lengths
	include 'pplv11inc/AXIS.INC'		! axis lengths
	include 'ferret_cmn/xrisc.cmn'		! to fix RISC FORTRAN bug
#else
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_SETUP.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE 'TMAP_PPLV11INC:AXIS_INC.DECL'	! axis lengths
	INCLUDE 'PPLV11INC:AXIS.INC'		! axis lengths
#endif

* PLOT+ parameter fom PPLV11INC:PARAMPL5.DAT
	INTEGER		nlabs
	PARAMETER     ( nlabs=50 )

* calling argument declarations:
	INTEGER		labnum, justify, xaxrel, yaxrel
	REAL		xpos, ypos, labht
	CHARACTER*(*)	string

* internal variable declarations:
	INTEGER		flen, lsym, ii
	CHARACTER	TM_FMT*4, font*3, labname(ppl_dset:ppl_offset)*6
	REAL 		xp, yp

* local parameter declarations:
	INTEGER		maxsym
	CHARACTER*1	esc
	PARAMETER     ( esc = CHAR(27),
     .			maxsym = 102   )! max. symbol length for PPLUS (PPL bug)

* data initialization
	DATA labname(ppl_dset)/'DSET'/,
     .	     labname(ppl_aspect)/'ASPECT'/,
     .	     labname(ppl_year)/'YEAR'/,
     .	     labname(ppl_datitl)/'DATITL'/,
     .	     labname(ppl_offset)/'OFFSET'/

* initialize fixed length string for RISC
	risc_buff = string
	len_rbuff = MIN( size_rbuff, LEN(string) )

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
	lsym = MIN( maxsym, LEN(string) )

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

* define the label to PPLUS
	IF     ( labnum .EQ. ppl_title ) THEN
* ... TITLE
	   WRITE ( ppl_buff, 3200) labht, font//risc_buff(:len_rbuff)
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	   WRITE ( ppl_buff, 3500 ) 'TIT', string(:lsym)
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	ELSEIF ( labnum .EQ. ppl_xlab ) THEN
* ... XLAB
	   CALL PPLCMD( from, line, 0,
     .			'XLAB '//font//risc_buff(:len_rbuff), 1, 1 )
	   WRITE ( ppl_buff, 3300 ) labht
	   CALL PPLCMD( from, line, 0, ppl_buff, 1, 1 )
	   WRITE ( ppl_buff, 3500 ) 'X', string(:lsym)
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	ELSEIF ( labnum .EQ. ppl_ylab ) THEN
* ... YLAB
	   CALL PPLCMD( from, line, 0,
     .			'YLAB '//font//risc_buff(:len_rbuff), 1, 1 )
	   WRITE ( ppl_buff, 3400 ) labht
	   CALL PPLCMD( from, line, 0, ppl_buff, 1, 1 )
	   WRITE ( ppl_buff, 3500 ) 'Y', string(:lsym)
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
#ifdef NO_SNGL
           risc_buff = TM_FMT(REAL(nlabs_on),3,3,flen)      ! for Unix/RISC
#else
           risc_buff = TM_FMT(SNGL(nlabs_on),3,3,flen)      ! for Unix/RISC
#endif
	   WRITE ( ppl_buff, 3500 ) risc_buff(:flen), string(:lsym)
	   CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
* ... define a PLOT+ symbol containing the label number "SET LABNUM_X 4"
*	or LABNUM_DSET, LABNUM_YEAR, etc.
	   ii = ppl_movlab - labnum
	   IF ( ii.GE.1 .AND. ii.LE.4 ) THEN
	      WRITE ( ppl_buff, 3500 )
     .			"num_"//ww_dim_name(ii), risc_buff(:flen)
	      CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	   ELSEIF ( ii.GE.ppl_dset  .AND. ii.LE.ppl_offset ) THEN
	      WRITE ( ppl_buff, 3500 )
     .			"num_"//labname(ii), risc_buff(:flen)
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
