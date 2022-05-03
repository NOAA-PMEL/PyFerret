	SUBROUTINE DISP_INIT( no_labels, cx_list, ncx )

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
* initialize the underlying plot package (PPLUS) for a fresh plot
* (this routine is not called for an overlayed plot)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 6/30/87 - extracted from DISP_SET_UP
* revision 0.1 - 7/21/87 - added XAXIS, YAXIS and WINDOW initialization
* revision 0.2 - 7/28/87 - removed LEV () to DISP_PREP, added documentation
* revision 0.3 - 8/26/87 - added TXTYPE YR
* revision 0.4 - 6/28/88 - added  AXLEN and AXSET
* revision 0.5 - 8/26/88 - removed SIZE
* revision 1.0 -  9/8/88 - removed all major re-setting to DISP_RESET 
* revision 1.1 - 1/26/89 - bug fix: 24-->13 for length of "date time" string 
* V200:   1/4/89 - major changes for plots with keys and joint contexts
* V230:   6/8/92 - added WIRE command (requiring WINDOW OFF) so set WINDOW dflt
*        7/22/92 - initialize ul_dolab so that /OVERLAYs following /NOLABEL
*                  will have predictable results
*        11/9/92 - unprotect colors if this is a fresh plot in a full window
*        1/28/93 - added args to plotify_grp_cx to promote thin regions to 2D
*       12/10/93 - set only the small tic size - do not mess with large tics
* V301:	 1/24/94 - turn on metafiles here instead of in XEQ_SET & START_PPLUS 
*	           (resolve problem of metafiles created with wrong name)
* V310:  3/21/94 - squeeze unneeded blanks from "FERRET Ver.3.10"
* v500:  1/26/99 - *kob* commented out PPLCMD call for aline, because this call
*                  was moved to disp_prep.F so reinitialization could occur for 
*                  plot overlays.
* v533:  6/27/01 - *acm* call PPLCMD to initialize AXTYPE,1,1 (not log axis) and
*		   initialize h_logaxis, v_logaxis to FALSE
* v533:  7/11/01 - *acm* call AXSET,1,1,1,1  to set to the default state of all axes on.
*      *acm* 10/01  remove VMS includes
* v540 *acm*  2/02 Do not call AXSET,1,1,1,1 - want PPL AXSET settings to persist
* V550 *acm* 12/02 use mode_labels and mode_logo to control labels and/or logo
* v552 *acm*  3/03 new param IAXSET to restore axes after PLOT/AXES/set plot.
* v552 *acm*  4/03 Delete symbols XAXIS_MIN, XAXIS_MAX, etc.  If they've
*                  been set by a previous plot, they dont get reset on plots
*                  like plot/vs xpts,ypts
* v580 *acm* 11/04 Fix bug 1009. Distinguish between axes which really have one 
*                  point, and those for which the hi and lo subscripts on the axes 
*                  are undefined. When undefined return zero from CX_DIM_LEN_ZERO in
*                  GET_CX_DIMS_ZERO. These axes are listed last in the heirarchy in 
*                  array plot_axis().  This means we can plot a variable with just 
*                  one point in their range (if /HLIMITS or /VLIMITS are used to give 
*                  PPLUS a range to draw).
* V600 *acm* 8/05 - Allow more digits in xrevision number
* V600 *acm* 5/06 - flag no_plot_yet for applying mode_linecolors setting ...
* V6.11 4/08 *acm* Fixes for bug 1571; call to PPL_AXES_RESTORE moved to DISP_INIT
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* PyFr *kms*  7/13 Scale default lengths by pplscale; 
*                  do not reset minor tic size (not scaled!)
* V686 *acm* 11/13 Allow symbol names to be up to 120 characters long
* V686 *acm* 12/13 set up the label symbols LABNUM_E and LABNUM_F
* V695+ *sh*  3/15 remove movable label rotations (left behind by FMRC annotations)
* V697 *acm* 12/15 Fixes for bug 2327 move call of PPL_AXES_RESTORE back to end of plot routines
* V697  *acm*  1/16 Ticket 2344: if there was a time-plot underlay, use its
*                   delta-t for time overlay plots. TAXUND restores the default.
* V697  *acm*  1/16 Ticket 2344: if there was a time-plot underlay, use the underlay's 
*                   time scaling for time overlay plots. TAXUND restores the default.
* V73+ 1/18 *acm*  Changes for Issue 1009; precsision in time info sent to pplus for 2D plots
*                  Initialize the information stored for this scaling
* V74+ 54/18 *acm*  Noticed that symbols LAB1, LAB6 etc labels persisted from 
*                   prev. plots, also LABKEY. Delete them
* V7.4+ *acm*  8/18 DSG-enabled Ferret: colorkey labels may be strings from the IDs
* V7.4+ *acm*  9/18 All moveable labels should be reset
* 4/22 *acm*  PyFerret issue #51: Allow for defining up to 4 more colors/pens to use in 
*                    graticule lines; vertical and horizontal, and possibly different styles
*                    for large and small tics. Custom colors/lines have been limited to 1, 
*                    so custom-color line plots with custom-color grats were not implemented.
*                    extrapen is 0 (for original pen); 1, 2, 3, 4 (graticule pens)

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'plot_setup.parm'
	include 'xplot_setup.cmn'
	include 'xplot_state.cmn'
	include 'xprog_state.cmn'
	include 'xrevision.cmn'
        include 'axis_inc.decl'    ! with iaxset
        include 'AXIS.INC'
	include 'xbox_bounds.cmn'
        include 'pyfonts.cmn'
        INCLUDE 'fgrdel.cmn'

* calling argument declarations:
	LOGICAL no_labels
	INTEGER ncx, cx_list(ncx)

* internal variable declarations:
	CHARACTER	TM_FMT*48, LEFINT*4, buffer*40, buffsym*30, show_str*12
        INTEGER         i, idim, ier, nc, slen
        REAL            logoht

* open metafile for this new plot
	CALL SEND_PLTYPE( mode_metafile )        ! added 1/94

* PLOT5 re-initialization (to forget previous states)
	CALL PPLCMD ( from, line, 0, 'XFOR', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'YFOR', 1, 1 )
 	CALL PPLCMD ( from, line, 0, 'XLAB', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'YLAB', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'XAXIS', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'YAXIS', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'LABS ', 1, 1 )
* 1/99 *kob* comment out call for aline as this was moved to disp_prep instead
*	CALL PPLCMD ( from, line, 0, 'ALINE ', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'TAXIS OFF', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'TAXUND 0', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'TITLE', 1, 1 )
*	CALL PPLCMD ( from, line, 0, 'TICS .125', 1, 1 )    ! chg 12/93
	CALL PPLCMD ( from, line, 0, 'WINDOW ON', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'AXTYPE 1,1', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'XVALOFF 0', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'YVALOFF 0', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'CLRKLAB', 1, 1 )

* Label numbers for the Ferret logo (if drawn) for pyFerret text settings
	
	DO i = 1, 3
	   logolab(i) = 0
	ENDDO

* clear rotations from first 50 moveable labels
	DO i = 1, 50
	   CALL PPLCMD ( from, line, 0, 'RLABS '//LEFINT(i,slen) , 1, 1)
	ENDDO

* Initialize extra-pen. 

	extrapen = 0

* if the axis min and max symbols already exist then delete them

        buffsym = 'XAXIS_MIN'
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'XAXIS_MAX'
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'YAXIS_MIN'
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'YAXIS_MAX'
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'X_REVERSED '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'Y_REVERSED '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)

* If LABNUM symbols exist from a previous plot, delete them.

        buffsym = 'LABNUM_X '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'LABNUM_Y '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'LABNUM_Z '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'LABNUM_T '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'LABNUM_E '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'LABNUM_F '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
           
        buffsym = 'LABNUM_DSET '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'LABNUM_ASPECT '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'LABNUM_YEAR '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'LABNUM_DATITL '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'LABNUM_OFFSET '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'LABNUM_DODS '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'LABNUM_CALEND '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'LABKEY '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)

        buffsym = 'LAB1 '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'LAB2 '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'LAB3 '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'LAB4 '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'LAB5 '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'LAB6 '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'LAB7 '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'LAB8 '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)
        buffsym = 'LAB9 '
	CALL GETSYM( buffsym, buffer, nc, ier )
	IF ( ier .EQ. 0 ) CALL DELSYM(buffsym,nc)


* free the color table on a full window
        IF ( vp_num .EQ. mvp_dflt ) CALL PPL_SHASET( 'RESET' )

        saved_bb_date = ' '
        bb_x_fac  = 1.
        bb_x_off  = 0.
        bb_y_fac  = 1.
        bb_y_off  = 0.

* initialize labels and keys
	nvect_on   = 0
	ncont_on   = 0
	nlabs_on   = 0
	nline_on   = 0
	nkeys_on   = 0
	novkeys_on = 0
	novkeys_ll = 0
	onedee_on  = .FALSE.
	twodee_on  = .FALSE.
	ovkey_ll_full = .FALSE.
	h_logaxis = .FALSE.
	v_logaxis = .FALSE.
        no_plot_yet = .FALSE.

        DO 200 idim = 1, nferdims
 200    ul_dolab(idim) = .TRUE.

* set up full context of plot (possibly multiple variables to plot)
	CALL GRP_CX_DOC( cx_list, ncx, cx_plot, key_doc )
	CALL GET_CX_DIMS_ZERO( cx_list(1), nplot_axis, plot_axis )
	CALL PLOTIFY_GRP_CX( cx_list, ncx )
	
        IF (.NOT.mode_labels ) no_labels = .FALSE.  ! Apply MODE LABELS setting
	
	IF ( no_labels ) RETURN

* set up positions for labels and keys  (including ul_dolab)
	CALL SETUP_KEYS( ncx )

* put program name and revision number onto plot
* ... "FERRET debug Ver. x.xx"   (or "FERRET Ver. 3.10"  3/94) 
        IF (mode_logo) THEN             ! Apply MODE LOGO setting
           show_str = TM_FMT(revision_level, 5, 12, slen)
	   IF ( progname_mod .NE. ' ' ) THEN
	      WRITE ( buffer, 3010) program_name(:len_program_name),
     .	  			 progname_mod(:len_progname_mod),
     .				 show_str(1:slen)
 3010	      FORMAT(A,1X,A,' Ver.',A)
	   ELSE
	      WRITE ( buffer, 3011) program_name(:len_program_name),
     .				 show_str(1:slen)
 3011	      FORMAT(A,' Ver. ',A)
	   ENDIF
           logoht = 0.5*dflt_letsize_label*textscale
	   CALL PUT_LABEL( ppl_movlab, 
     .                     0.0, 1.0*pplscale + 3.6*logoht,
     .                     ppl_centered, logoht,
     .                     buffer(:len_program_name+len_progname_mod+11),
     .                     lab_loc_axlen_plus, lab_loc_axlen_plus )
	   logolab(1) = nlabs_on
* ... "NOAA/PMEL TMAP"
	   CALL PUT_LABEL( ppl_movlab, 
     .                     0.0, 1.0*pplscale + 1.8*logoht,
     .                     ppl_centered, logoht,
     .                     'NOAA/PMEL TMAP',
     .                     lab_loc_axlen_plus, lab_loc_axlen_plus )
	   logolab(2) = nlabs_on
* ... "8-SEP-1988 11:05"
	   CALL PUT_LABEL( ppl_movlab, 
     .                     0.0, 1.0*pplscale,
     .                     ppl_centered, logoht,
     .                     '''date'' ''time''',
     .                     lab_loc_axlen_plus, lab_loc_axlen_plus )
	   logolab(3) = nlabs_on
        ENDIF ! mode_logo

	RETURN
	END
