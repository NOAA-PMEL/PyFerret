	LOGICAL FUNCTION SCALAR_GRID_LINE( idim, grid, cat, var)
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

* determine if the indicated axis is a single point axis
* meaning line_dim of 1 or NORMAL, or user variable compressed to a point

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - TMAP 
*
* V698 3/16 *sh*

* arguments:
* inputs:
*	idim	  - dimension to test (input)
*	grid	  - grid to test (input)
*	cat	  - category of variable to test (input)  (only uvars matter)
*	var	  - (user) variable to test (input)

      IMPLICIT NONE
	include 'netcdf.inc'
#include "tmap_dset.parm"
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'xtm_grid.cmn_text'  
	include 'xvariables.cmn'

* calling argument declarations:
	INTEGER idim, grid, cat, var

* local variable declarations:
	INTEGER line

* initialize
	line = grid_line(idim, grid)

* is the axis unit length
	IF (line .EQ. mnormal
     .	.OR.line_dim(line) .EQ. 1) GOTO 1000

* is this a user variable compressed to a point
	IF (cat .EQ. cat_user_var) THEN
	   IF (uvar_given(idim,var).EQ.uvlim_gvn_xact) GOTO 1000
	ENDIF

* not a unity length axis
	SCALAR_GRID_LINE = .FALSE.
	RETURN

* yes, a unity length line
 1000	SCALAR_GRID_LINE = .TRUE.
	RETURN

	END
