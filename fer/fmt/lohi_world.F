	CHARACTER*(*) FUNCTION LOHI_WORLD( idim,cx,to_word,full_prec,slen )

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
* create a string describing the world coordinate boundaries along an axis
* for example: 160E:140W or 10m (if lo depth = hi)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/26/87
* revision 0.1 - 2/25/88 - changed TRANSLATE_TO_WORLD call to allow non-lat/long
*			   formatting.
* revision 0.2 - 3/25/88 - number of decimals set in SET MODE command
*			   and possibly reverse ordering of time limits since
*			   stored times may be tsteps (for a particular T axis)
*			   or -1*(secs_from_BC)  (independent of partic. axis)
* revision 0.3 - 8/12/88 - negative times not converted to time steps
* revision 1.0 -  1/10/89- split off actual string creation into LOHI_STRING
* V200:  5/19/89 - 4D symmetrical - point limits have ww coords at box edges 
*	 3/22/90 - for cx.LE.cx_last use last successful grid
*	 9/11/90 - corrected bug: IF test treated cx_buff as a stored region
* V300:   2/3/93 - the 9/11/90 bug fix prevents the default region for being
*                  formatted

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xvariables.cmn'		! added 3/90
	include 'ferret_cmn/xfr_grid.cmn'		! added 3/90
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'		! added 3/90
	INCLUDE 'FERRET_CMN:XFR_GRID.CMN'		! added 3/90
#endif

* calling argument declarations
	LOGICAL	full_prec
	INTEGER	idim, cx, slen
	CHARACTER*(*) to_word

* internal variable declarations:
	INTEGER	sub_len, ndec, grid, status, cx_try
	REAL*8	TM_WORLD, bc_to_t0, lo, hi, temp
	CHARACTER*40	LOHI_STRING

* grid number is used to determine formatting option
* ( cx<0 for stored (named) regions )  (cx=cx_last for default region)
* 2/93 - treat cx_last like stored regions
* 2/93 - expanded the logic for finding **some** grid to use when their
*        isn't one - search up the context stack
        IF ( cx .LE. 0 .OR. cx .EQ. cx_last ) THEN
           cx_try = is_cx( 1 )
           IF ( cx_try.EQ.0 .OR. cx_try.EQ.unspecified_int4 ) THEN
              grid = unspecified_int4
           ELSE
              grid = cx_grid( cx_try )
           ENDIF
* ... desperation search added 2/3/93
* don't know why is_cx(1) isn'r reliable (havn't checked)
           IF ( grid .EQ. unspecified_int4 ) THEN
              DO 100 cx_try = cx_cmnd+1, cx_cmnd+5   ! arbitrary limit
                 grid = cx_grid( cx_try )
                 IF ( grid.NE.0 .AND. grid.NE.unspecified_int4) GOTO 200
 100          CONTINUE
           ENDIF
        ELSE
           grid = cx_grid( cx )
        ENDIF
	IF ( grid .EQ. unspecified_int4 ) grid = mgrid_abstract

* is this just a single grid box with the world coord bod edges given ?
 200    IF ( cx_lo_ss(cx,idim) .NE. unspecified_int4
     .	.AND.cx_lo_ss(cx,idim) .EQ. cx_hi_ss(cx,idim) ) THEN
	   lo = TM_WORLD( cx_lo_ss(cx,idim),grid, idim, box_middle )
	   hi = lo
	ELSE
	   lo = cx_lo_ww( idim, cx )
	   hi = cx_hi_ww( idim, cx )
	ENDIF

* determine precision level for output string
	IF ( full_prec ) THEN
	   ndec = ax_dec_pt( idim )
	ELSE
	   ndec = 0
	ENDIF

* create the string
	LOHI_WORLD = LOHI_STRING( lo,hi,idim,grid,to_word,ndec,slen )
	
	RETURN
	END
