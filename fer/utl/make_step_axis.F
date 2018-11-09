	SUBROUTINE MAKE_STEP_AXIS( grid, idim, inc, lo_ss, hi_ss, axis )

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
* create a REAL*4 array of world coordinates over a range of subscripts along
* axis IDIM of grid GRID.  Begin the axis at subscript 1 of AXIS.
* If argument "inc" is 2 then output lo/hi limits for each point
* If "inc" is 3 then do lo/hi limits and duplicate upper limit

* programmer - steve hankin
* V530 *sh* 1/01 
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
* V698  *acm* 5/16 Ticket 1845, 2404 fix; adjusting plot-axis ends for subspan 
*		   modulo axes.

        IMPLICIT NONE
	include 'ferret.parm'
	include 'tmap_dims.parm'
        include 'xtm_grid.cmn_text'

* calling argument declarations:
	INTEGER	grid, idim, inc, lo_ss, hi_ss
	REAL    axis(*)

* internal variable declarations:
	LOGICAL TM_ITS_SUBSPAN_MODULO, is_subspan, void_at_start, void_at_end
	INTEGER	ito, i, iaxis
	REAL*8	TM_WORLD

	ito = 0
	iaxis = grid_line(idim,grid) 
	
        is_subspan = TM_ITS_SUBSPAN_MODULO( iaxis  )
        void_at_start = is_subspan .AND. (lo_ss.EQ.0)
        void_at_end  = is_subspan .AND. (hi_ss.EQ.line_dim(iaxis)+1 )

	IF (inc .EQ. 1) THEN
* make a simple axis - 1 output point per axis point
	   DO i = lo_ss, hi_ss
	      ito = ito + 1
	      axis(ito) = TM_WORLD( i, grid, idim, box_middle )
	   END DO
	   IF (is_subspan) THEN
	      IF (void_at_start) axis(1) = TM_WORLD( lo_ss, grid, idim, box_hi_lim )
	      IF (void_at_end) axis(ito) = TM_WORLD( hi_ss, grid, idim, box_lo_lim )
	   ENDIF

	ELSEIF (inc .EQ. 2) THEN
* make a 2-point "step" axis - output lo and hi points at each axis point
	   DO i = lo_ss, hi_ss
	      axis(ito+1) = TM_WORLD( i, grid, idim, box_lo_lim )
	      axis(ito+2) = TM_WORLD( i, grid, idim, box_hi_lim )
	      ito = ito + 2
	   END DO
	   IF (is_subspan) THEN
	      IF (void_at_start) axis(1) = TM_WORLD( lo_ss, grid, idim, box_hi_lim )
	      IF (void_at_end) axis(ito) = TM_WORLD( hi_ss, grid, idim, box_lo_lim )
	   ENDIF
	ELSE
* make a 3-point "step" axis - lo then hi twice
	   DO i = lo_ss, hi_ss
	      axis(ito+1) = TM_WORLD( i, grid, idim, box_lo_lim )
	      axis(ito+2) = TM_WORLD( i, grid, idim, box_hi_lim )
	      axis(ito+3) = axis(ito+2)
	      ito = ito + 3
	   END DO
	   IF (is_subspan) THEN
	      IF (void_at_start) axis(1) = TM_WORLD( lo_ss, grid, idim, box_hi_lim )
	      IF (void_at_end) axis(ito) = TM_WORLD( hi_ss, grid, idim, box_lo_lim )
	      IF (void_at_end) axis(ito-1) = axis(ito+hi_ss*3)
	   ENDIF
	ENDIF

	RETURN
	END
