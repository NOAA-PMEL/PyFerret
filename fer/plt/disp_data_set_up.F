	SUBROUTINE DISP_DATA_SET_UP(  ax1_work, ax2_work, status )

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
* set up graphics environment and obtain the data

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 1/7/87 
* revision 0.1 - 2/25/87 - changes for INTERPRETATION STACK
* revision 0.2 - 6/24/87 - changes to accomodate transposed plots
* revision 0.3 - 9/16/87 - added want_cmnd_data to CALL GET_CMND_DATA
* V200:  5/19/89 - 4D symmetrical
*         5/9/90 - allow CGRID_SIZE for axes to accomodate PLOT/VS
* V301:   1/7/94 - don't unprotect the command data cuz' GET_MEMORY is called
*		   during set-up of plots
* V312: 5/94 - array "memory" as a calling argument
* V500: 2/99 *sh* - added support for POLYGON command -- working storage needs
*       2/99 *sh* - bug fix: plot working storage based on cx(1) only
*			can overflow for PLOT v1,v2  if v2 is bigger
* V530: *sh* 9/00 - added initial data type support in get_cmnd_data 
*	*sh* 1/01 - added PLOT/STEP with special work array needs
*	          - also fixed apparent bug in loop over num_uvars_in_cmnd
* V533: *sh* 6/01 - include CGRID_SIZE in determination of axis buffer sizes
*	            only if PLOT/VS
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* V69+ *acm*  9/14 Ticket 1758. Memory request uses number of blocks not words
* V702 1/17 *sh* for trac enhancement #2369 -- dynamic memory management 
* V73+ *acm* 12/17 additions for plotting several lines from a dsg dataset
* V74+ *acm*  4/18 additions for plotting trajectories from a dsg dataset
* V74+ 4/18 *acm*  Automatically do a colored trajectory plot for the command 
*                  "PLOT var" if it's a DSG trajectory set. Allow color-by-Feature-id.
* V74+ 7/18 *acm*  DSG trajectory plots colored by ID even if its a string 
* V75+ 5/19 *acm*  DSG lon/lat plots of profile or timeseries datasets
* V76  1/20 *acm* working with Point-type dsg data
* V760 *acm* 3/20 Flag for case when constraints result in no features
* v763 *acm* 9/20 Plots of id-variables in trajectory-profile, timeseries-profile data

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'slash.parm'
	include 'command.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xdsg_context.cmn'
	include 'xprog_state.cmn'
#include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	include 'xtm_grid.cmn_text'
	include 'xtext_info.cmn'

	
* calling argument declarations:
	INTEGER	ax1_work, ax2_work, status

* internal variable declarations:
	LOGICAL TM_ITSA_DSG_RAGGED, TM_ITSA_DSG, 
     .		plot_vs, typerr, its_dsg, its_cmpnd, its_traj, 
     .		dsg_as_traj, dsg_as_time
	INTEGER	MR_DIM_LEN, CX_DIM_LEN, CGRID_SIZE, TM_DSG_NFEATURES,
     .		TM_DSG_DSET_FROM_GRID, DSG_WHATS_IT,
     .		idim, ndim, dim(nferdims), mr1, cx, ivar,
     .		factor, loc, grid, dset, nfeatures, obsdimlen, 
     .		fline, dtyp, numv, along_dim
	INTEGER*8 i8_ax_len
	REAL	big_val, little_val
	CHARACTER buff*4, buff1*1

* check for PLOT/VS -- significant in axis buffer sizes
	plot_vs = cmnd_num .EQ. cmnd_plot
     .	          .AND. qual_given(slash_plot_vs) .GT. 0

* initialize
	cxdsg_empty_set = .FALSE.

* get the requested data somehow  (and leave it protected 1/94)

	dset  = cx_data_set(cx_last)
	dtyp = ptype_float

	IF (dset .GT. pdset_irrelevant) THEN
	   IF (dsg_ragged(dset)) dtyp = ptype_native
	ENDIF
	CALL GET_PROT_CMND_DATA (  cx_last, dtyp, status )

	IF ( status .NE. ferr_ok ) GOTO 5100

	numv = num_uvars_in_cmnd


* ragged DSG data?
* For a trajectory dataset, the variable may be on the instance axis 
* (color by feature-id). If so set that up. Otherwise insist it's a
* variable on the obs axis of a ragged-array dataset

         cx = is_cx(1)
	 grid = cx_grid(cx)
	 
	 grid_is_dsg = TM_ITSA_DSG( grid )
	 IF (DSG_WHATS_IT(grid) .EQ. pdsg_fs_dim) THEN
	    grid_is_dsg = .TRUE.
	    grid = dsg_xlate_grid(dset)
	 ENDIF

*  IF (PLOT/ALONG=xy/z=lo:hi ) for trajectoryprofile --> dsg_as_traj
*  TODO: 
*    Trajectory as timeseries              

	 IF (cmnd_num .EQ. cmnd_plot .AND. grid_is_dsg) THEN

	    dset = TM_DSG_DSET_FROM_GRID( grid )
	    nfeatures = TM_DSG_NFEATURES( grid )	    
	    CALL TM_DSG_FACTS( grid, idim, obsdimlen, fline, its_dsg, its_cmpnd )

	    CALL DSG_PLTALONG_SETUP (dset, idim, its_traj, dsg_as_traj, 
     .				dsg_as_time, status)
	    IF (status .NE. ferr_ok) GOTO 5200

	    IF (idim.EQ.pfeatureType_Trajectory .OR. idim.EQ.pfeatureType_Point) plot_vs = .TRUE.
	    IF (.NOT.plot_vs) THEN
	       typerr = (cx_type(cx) .EQ. ptype_string) 
	       IF (typerr) GOTO 5200
	       IF (grid_is_dsg) CALL PLOT_DSG_DATA_SET_UP(nfeatures, obsdimlen, 
     .				 dsg_as_time, status)
	       IF (status .NE. ferr_ok) GOTO 5200
            ENDIF
	 ENDIF

* PLOT/ALONG -- reinterpret a multi-D data block as a collection of lines
	IF (  cmnd_num .EQ. cmnd_plot
     .	.AND. qual_given(slash_plot_along) .GT. 0
     .	.AND. .NOT.plot_vs   .AND. .NOT. grid_is_dsg  ) THEN
	   CALL PLOT_ALONG_DATA_SET_UP(status)
	   IF (status .NE. ferr_ok) GOTO 5010
	ENDIF

* PLOT/STEP needs double or triple the working storage
*    double to make connected steps, triple for disconnected
	factor = 1
	IF (cmnd_num .EQ. cmnd_plot) THEN
	   loc = qual_given( slash_plot_step )
	   IF ( loc .GT. 0  ) THEN
	      CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                           buff, status )
              IF ( status .NE. ferr_ok ) RETURN
	      IF ( buff(1:3) .EQ. 'con' ) THEN
	         factor = 2
	      ELSE
	         factor = 3
	      ENDIF
	   ENDIF	   
	ENDIF

* compute working storage
	IF (cmnd_num .EQ. cmnd_polygon) THEN
* ... allocate buffers used to extract individual polygon coord arrays
*     set buffer size to the longest axis of 1st coord arg
*     Polygon may need extra working storage to store info about 
*     missing polygons on a time axis
	   cx = is_cx(1)
	   CALL GET_CX_DIMS(cx, ndim, dim)
	   i8_ax_len = 1
	   DO 100 idim = 1, ndim
 100	   i8_ax_len = 2* MAX(i8_ax_len,
     .			cx_hi_ss(cx,dim(idim)) - 
     .			cx_lo_ss(cx,dim(idim)) + 1 )
	ELSE
* ... allocate memory used for axes
*     make them each equal to MAX in case /TRANSPOSE or mulit-dim PLOT/VS
* (2/99: loop over all arguments to find largest requirement)
* ?? why is CGRID_SIZE needed ... could get very large ??
	   i8_ax_len = 1
	   DO 200 ivar = 1, numv
	      mr1 = is_mr( ivar )
	      CALL GET_CX_DIMS( is_cx(ivar), ndim, dim )
	      i8_ax_len = MAX( i8_ax_len,
     .			    MR_DIM_LEN( dim(1), mr1 ),
     .		            MR_DIM_LEN( dim(2), mr1 ),
     .			    factor*CX_DIM_LEN(dim(1), is_cx(ivar)),
     .			    factor*CX_DIM_LEN(dim(2), is_cx(ivar)) )
	      IF (plot_vs) i8_ax_len = MAX( i8_ax_len,
     .		            CGRID_SIZE(is_cx(ivar))  )
 200	   CONTINUE

* trajectory plot from DSG trajectory file needs obs-length 
* (color var may be feature-length).

	   IF (cmnd_num.EQ.cmnd_plot .AND. grid_is_dsg ) THEN
              i8_ax_len = obsdimlen
	   ENDIF
	ENDIF

* allocate working storage
* the somewhat odd logic is cuz we're making minimal changes to the 
* pre-dynamic memory code
	CALL CREATE_DYN_WRKSPC( i8_ax_len, ax1_work, status )
	IF ( status .NE. ferr_ok ) GOTO 5020
	CALL CREATE_DYN_WRKSPC( i8_ax_len, ax2_work, status )
	IF ( status .NE. ferr_ok ) GOTO 5010

* successful completion
	RETURN

* error exits - release the memory used for axis buffers
 5010	CALL FREE_WS_DYNMEM( ax1_work )
 5020	CALL UNPROTECT_CMND_DATA	! added 2/99
 5100	RETURN

 5200	CONTINUE
* ... copied from get_prot_cmnd_data error exit ... 
        CALL RELEASE_DYN_WORK_SPACE
        CALL DELETE_MRS_IN_PROGRESS
        CALL UNPROTECT_ALL_MRS
        CALL DEALLO_ALL_GRIDS
        CALL CLEAN_STACK
	buff = ' '

	IF (typerr) CALL ERRMSG( ferr_data_type, status,buff, *5100 )

	END
