	SUBROUTINE DISP_RESET

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
* initialize or reset the underlying plot package (PPLUS) to FERRET defaults
* (this routine is called only at start-up or by the command PPLUS/RESET)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/8/88 - extracted from DISP_INIT
* V200:   1/3/89 - added ALINE, AXLSZE
* V230:  11/4/92 - call SET_VIEWPORT(dflt_vp)
*        11/9/92 - unprotect all colors
* V540:  *sh* 9/01 - grab the PLOT+ origin from COMMON
* V600 *acm* 5/2006 LINECOLORS User can change the number of line colors.
* V62  *acm* 4/09 - Restore default settings for AXTYPE, AXLABP, AXLINT,
*                   TXLABP, TXLSZE, TICS, CONPRE, CONPST, DFLTFNT, palette,
*                   Default number of colors. (Should reset the line colors??)
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus.
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* PyFr *kms*  7/13 reset to scaled lengths and sizes using SET_PPL_VALUES
* v696  *acm* 6/15 PLOT/VS/GAPLOC to define seperate line segments on
*                  PLOT/VS plots (LAS ticket 1597 and others)
* V697  *acm*  1/16 Ticket 2344: if there was a time-plot underlay, use the underlay's 
*                   time scaling for time overlay plots. TAXUND restores the default.
* V7.4+ *acm*  8/18 DSG-enabled Ferret: colorkey labels may be strings from the IDs
* V764 11/21 *acm* For PyFerret issue 98, if PLOT/SET/PALETTE, save info so the color 
*                 palette can be restored after the plot is completed. Initialize the flag here.

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'plot_setup.parm'
	include 'xplot_setup.cmn'
	include 'xplot_state.cmn'
        include 'xprog_state.cmn'
	include 'xrevision.cmn'
        include 'gkscm1_inc.decl'
        include 'GKSCM1.INC'   ! for wsid

* local variable declarations
	CHARACTER buff*24

* PLOT5 re-initialization (to forget previous states)

*       set the default plot size
        wn_xinches(wsid) = dflt_xinches
        wn_yinches(wsid) = dflt_yinches

*       set sizes and locations that scale with the plot size
        CALL SET_PPL_VALUES(wsid, .TRUE.)

	CALL PPLCMD ( from, line, 0, 'XFOR', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'YFOR', 1, 1 )
 	CALL PPLCMD ( from, line, 0, 'XLAB', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'YLAB', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'XAXIS', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'YAXIS', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'XVALOFF 0', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'YVALOFF 0', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'AXSET 1,1,1,1', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'PEN 0,1', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'PEN 1,1', 1, 1 )

        WRITE (buff, 100) mode_arg(pmode_linecolors, 1)
  100   FORMAT (I3)
        CALL PPLCMD (from, line, 0, 'LINECOLORS '//buff(1:3)//' 1', 1, 1)

	CALL PPLCMD ( from, line, 0, 'AXATIC 10,10', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'LABS', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'ALINE', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'TAXIS OFF', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'AXNMTC 0,0', 1, 1 )
c	CALL PPLCMD ( from, line, 0, 'TXNMTC 0', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'TXNMTC -1', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'TAXUND 0', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'TITLE', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'WINDOW ON', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'BOX OFF', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'CROSS 0', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'AXTYPE 1,1', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'AXLABP -1,-1', 1, 1)
	CALL PPLCMD ( from, line, 0, 'TXLABP -1', 1, 1)
	CALL PPLCMD ( from, line, 0, 'AXLINT 2,2', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'CONPRE ', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'CONPST ', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'GAPLOC 0', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'CLRKLAB', 1, 1 )

* remove viewports and clear window, restore axis sizes, update window
        CALL SET_VIEWPORT( mvp_dflt ) 

* set up the default spectrum
        CALL PPL_SHASET( 'RESET' )
        CALL PPL_SHASET( 'SPECTRUM=default' )
	set_palette = .FALSE.

* Set default (initial) number of color levels

        CALL SET_DEFAULT_NLEVELS( mode_arg(pmode_nlevels, 1) )

	RETURN
	END
