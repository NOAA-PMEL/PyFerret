	REAL FUNCTION BOX_SIZE( isubscript, grid, idim )

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
* given a subscript value on a grid and an axis return the length of the
* grid box

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 12/9/86
* revision 1.0 - 3/10/87 - line data stored in COMMON/~VARIABLES/
* revision 0.2 - 3/23/87 - added new TMAP library: time axes, file var. names
* revision 0.3 - 2/18/88 - incorporated "regular" lines
* revision 1.0 - 1/12/88 - new TM IO library stores box bounds, not sizes
* Unix/RISC port - 1/91 - cant use "/star" in documentation
*                  9/91 - incorporated (missing) modulo axis logic
* V4.20 - 10/95 - remove references to line_mem to permit recursive line defs
*	   1/96 - bug fix: only look at line_parent for dynamic lines
*		           (code would be sweeter if ALL lines had line_parent)
* V510 4/00 *sh -- all lines now DO have line_parent

#ifdef unix
	include 'tmap_dims.parm'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret.parm'
	include	'xvariables.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER	isubscript, grid, idim

* internal variable declarations:
	LOGICAL recursive
	INTEGER	iaxis, isub0
	REAL*8	TM_WORLD, lo_point, hi_point

* initialize
	iaxis = grid_line( idim, grid )
	IF ( iaxis .EQ. mnormal .OR. iaxis .EQ. munknown ) THEN
	   BOX_SIZE = bad_val4
	   RETURN
	ENDIF

	IF ( line_regular( iaxis ) ) THEN
* regularly spaced points all have same grid box size
	   BOX_SIZE = SNGL( line_delta( iaxis ) )

	ELSE
* irregularly spaced points
* keep subscript within bounds ( between 1 and dimension of axis ) by ...
           IF ( line_modulo( iaxis ) ) THEN
* ... modulo-ing to the appropriate value
              isub0 = MOD(isubscript-1,line_dim(iaxis)) + 1
           ELSE
* ... truncating to the appropriate value
	      isub0 = MIN( line_dim(iaxis), MAX( 1, isubscript ) )
           ENDIF

* only dynamic lines have the possibility of recursive definitions
* (since FORTRAN doesn't protect "AND" clauses this test in 2 steps)
	   recursive = line_parent(iaxis) .NE. 0 

* use TM_WORLD to compute BOX_SIZE for recursive defs (10/95)
	   IF ( recursive ) THEN
* ... every nth point from an irregular axis --> use midpoints as box bounds
	      IF (isub0.EQ.1 .AND. .NOT.line_modulo(iaxis) ) THEN
	         lo_point =  TM_WORLD(isub0,   grid, idim, box_lo_lim)
	      ELSE
	         lo_point = (TM_WORLD(isub0,   grid, idim, box_middle) 
     .			  +  TM_WORLD(isub0-1, grid, idim, box_middle)) / 2
	      ENDIF
	      IF ( isub0.EQ.line_dim(iaxis) 
     .	     .AND. .NOT.line_modulo(iaxis) ) THEN
	         hi_point =  TM_WORLD(isub0,   grid, idim, box_hi_lim)
	      ELSE
	         hi_point = (TM_WORLD(isub0,   grid, idim, box_middle) 
     .			  +  TM_WORLD(isub0+1, grid, idim, box_middle)) / 2
	      ENDIF
	      BOX_SIZE = SNGL(hi_point - lo_point)
	   ELSE
	      BOX_SIZE = SNGL( TM_WORLD(isub0, grid, idim, box_hi_lim)
     .		             - TM_WORLD(isub0, grid, idim, box_lo_lim) )
	   ENDIF

	ENDIF

	RETURN
	END
