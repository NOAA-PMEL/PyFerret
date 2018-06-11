	SUBROUTINE CONFORMABLE_AGG_GRID (agg_dim, vname, imemb,
     .					 res_grid, res_cat, res_var,
     .					 com_grid, com_cat, com_var,
     .					 orig_grid, status )

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
* test that the grid of a member of an aggregation is conformable with the
* grid of the aggregation result.
* scalar axes in the result must correspond to scalar axes in the member
* axes with ranges must be the very same axis ... and not compressed
* success or failure is indicated by the return status

* V698 3/16 *sh*
* V741 *acm* 5/18 Promoting global attributes to a variable in ensembles

	include	'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'
	include	'xtext_info.cmn'
	include 'xtm_grid.cmn_text'   ! for grid_line
	include 'xfr_grid.cmn'

* calling argument declarations:
	INTEGER res_grid, res_cat, res_var, agg_dim,
     .		com_grid, com_cat, com_var, orig_grid, imemb, status
	CHARACTER*(*) vname

* internal variable declarations
	LOGICAL SCALAR_GRID_LINE
	INTEGER	TM_LENSTR1, idim, slen, vlen
	CHARACTER LEFINT*6

	IF (com_grid .EQ. mgrid_abstract) GOTO 4000

	DO idim = 1, nferdims
	   IF (idim .EQ. agg_dim) THEN
	      IF (.NOT.SCALAR_GRID_LINE(idim, com_grid, com_cat, com_var)) GOTO 5100
	   ELSEIF (agg_dim.EQ.F_dim .AND. idim.EQ.T_dim) THEN
	      IF (grid_line(idim,com_grid) .NE. grid_line(idim,orig_grid))
     .						      GOTO 5100
	   ELSEIF (SCALAR_GRID_LINE(idim, res_grid, res_cat, res_var)) THEN
	      IF (SCALAR_GRID_LINE(idim, com_grid, com_cat, com_var)) THEN
* ... single point component axis need not match result context
*     e.g. Z=0:500@ave on component may map to Z=250 single point result 
	         CYCLE
	      ELSE
	         GOTO 5100
	      ENDIF
	   ELSE
	      IF (grid_line(idim,com_grid) .NE. grid_line(idim,res_grid))
     .							GOTO 5100
* ... a LET/D compressed to a point may share axis but still be nonconformable
	      IF (SCALAR_GRID_LINE(idim, com_grid, com_cat, com_var)) GOTO 5100
	   ENDIF
	ENDDO

* success
 4000	status = ferr_ok
	RETURN

* error exit
 5000	RETURN
 5100	vlen = TM_LENSTR1(vname)
	CALL ERRMSG(ferr_aggregate_error, status,
     .		    'grid definition of '// vname(:vlen)//
     .		    ' has changed on axis '//ww_dim_name(idim)//
     .		    ' of agg member '//LEFINT(imemb,slen), *5000)
	END	
