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

	include 'ferret.parm'
	include 'plot_setup.parm'
	include 'xplot_setup.cmn'
	include 'xplot_state.cmn'
        include 'xprog_state.cmn'
	include 'xrevision.cmn'

* local variable declarations
	CHARACTER buff*24
	LOGICAL	  ITS_BATCH_GRAPHICS, batch

* local parameters
        LOGICAL     physical
        PARAMETER ( physical = .TRUE. )

* PLOT5 re-initialization (to forget previous states)
!	CALL PPLCMD ( from, line, 0, 'GET FERRET_PPLINIT', 1, 1 )
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

	buff = ' '
	WRITE ( buff, 3000 ) xwhitelo, ywhitelo
 3000	FORMAT(F9.3,F9.3)
	CALL PPLCMD ( from, line, 0, 'ORIGIN '//buff, 1, 1 )
	CALL PPLCMD ( from, line, 0, 'AXATIC 10,10', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'LABS', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'ALINE', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'TAXIS OFF', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'AXNMTC 0,0', 1, 1 )
c	CALL PPLCMD ( from, line, 0, 'TXNMTC 0', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'TXNMTC -1', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'TITLE', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'LABSET,0.16,0.12,0.12,0.12', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'WINDOW ON', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'BOX OFF', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'CROSS 0', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'AXLSZE .1,.1', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'AXTYPE 1,1', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'AXLABP -1,-1', 1, 1)
	CALL PPLCMD ( from, line, 0, 'TXLABP -1', 1, 1)
	CALL PPLCMD ( from, line, 0, 'TXLSZE .1', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'TICS,.125,.25,.125,.25,-1,-1', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'AXLINT 2,2', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'CONPRE ', 1, 1 )
	CALL PPLCMD ( from, line, 0, 'CONPST ', 1, 1 )

	CALL SET_PLOT_SIZE( 10.2, 8.8 ) ! initial size
        CALL SET_VIEWPORT( mvp_dflt )   ! remove viewports, restore axis sizes

c        CALL COLOR (1)  ! ppl is not always in the right state to call this...

* set up the default spectrum
        CALL PPL_SHASET( 'RESET' )
        CALL PPL_SHASET( 'SPECTRUM=default' )

* Set default (initial) number of color levels

        CALL SET_DEFAULT_NLEVELS( mode_arg(pmode_nlevels, 1) )

	RETURN
	END
