	SUBROUTINE RESET_DSG (nvars, cx_list)

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
* Clean up after a DSG plot; extra lines and grids were alloocated.
* Reset cx_feature, the number of variables plotted, cx_grid
* If a mask variable was used the number of variables and variable-
* memory pointer needs restoring
* V760 *acm* 3/20 Flag for case when constraints result in no features
*                 Store plot state for axlab position.


	include	'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xtm_grid.cmn_text'
	include 'xprog_state.cmn'
	include 'xdsg_context.cmn'

	INTEGER nvars, cx_list(*)

	LOGICAL its_dsg, its_cmpnd
	INTEGER TM_DSG_DSET_FROM_GRID, CX_DIM_LEN,
     .		cx, grid, dset, i, the_cx, iaxis, orientation, 
     .		obsdimlen, fline, nv, status
	CHARACTER buff*32, TM_FMT*48, fmt*6

* Get the DSG dataset, grid, and the orientation of the feature
* so we can get the axes that were allocated.
* If plotting timeseriesProfile as a set of time series, the 
* directions of the grids has been changed.  Recover the information
* needed to reset using cx_buff

	cx = is_cx(1)
	grid = cx_grid(cx)
	dset = TM_DSG_DSET_FROM_GRID( grid )

	IF (dset .LE. pdset_irrelevant) THEN
	   grid = cx_grid(cx_buff)
	   dset = TM_DSG_DSET_FROM_GRID( grid )
	ENDIF
	IF (dset .LE. pdset_irrelevant) GOTO 5100

	nv = nvars
	
	its_dsg = .TRUE.
	CALL TM_DSG_FACTS( grid, orientation, obsdimlen, fline, its_dsg, its_cmpnd )

	IF (nv .GT. 1) THEN

* Was this a single-feature, multi-variable plot?  If so we defined one
* temporary grid.
	   IF (cx_variable(cx) .NE. cx_variable(cx+1) ) THEN 
	       nv = 1
	   ELSE

* Restore number of variables - num_uvars_in_cmnd is reset for multi-
* feature plot or for mask-variable.
	      nvars = 1

	   ENDIF
	ENDIF

* Deallocate the temporary grid and axis used.  Reset cx_feature.
	
	the_cx = cx - 1
	DO i = 1, nv
	   the_cx = the_cx + 1
	   grid = cx_grid(the_cx)
	   iaxis = grid_line(orientation, grid)
	   
	   IF (iaxis .EQ. 0 .AND. its_cmpnd) THEN
	      iaxis = grid_line(t_dim, grid)
	      IF (iaxis .EQ. 0) iaxis = grid_line(z_dim, grid)
	   ENDIF

	   IF (iaxis .EQ. 0) CYCLE

	   CALL TM_deallo_dyn_grid( grid )
	   CALL FREE_LINE_DYNMEM ( iaxis )

           cx_feature(cx_list(i)) = pdsg_not_dsg  ! zero
	ENDDO

	grid_is_dsg = .FALSE.
	cx_grid(cx) = cx_grid(cx_buff)

* Flag for dsg context that resulted in no data.
* Restore plot state for axlab position to previous setting.

	IF (cxdsg_empty_set) THEN
	   CALL PPLCMD ( ' ', ' ', 0, 'YAXIS', 1, 1 )
	   CALL PPLCMD ( ' ', ' ', 0, 'AXLABP,'//cxdsg_axlabp_save, 1, 1) 
	ENDIF

	cxdsg_empty_set = .FALSE.

 5000	RETURN
 5100	CALL ERRMSG( ferr_internal, status, 'RESET_DSG', *5000)

	END
