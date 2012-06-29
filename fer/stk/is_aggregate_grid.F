	SUBROUTINE IS_AGGREGATE_GRID(uvar, dset, cx, status )

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
* examine members of an aggregate datset, getting grids for 
* elements that are user-defined (LET/D) variables. Use the 
* interpretation stack in order to determine a defining grid for the
* overall aggregate dataset.
* Detailed description of the underlying logic is in MERGE_CONTEXT
* Note: When pseudo-variables are used (eg. LIST U/Z[K=1:3]) the grids for them
*	may not be known until all other components are examined.  In this
*	case a 2nd pass is required

* programmer - Ansley Mankek (from is_uvar_grid)
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*

* calling argument declarations:
* uvar and dset point to the variable in the aggregate dataset
* And cx_orig points to the context for that variable.

	INTEGER	uvar, dset, cx, status

* internal variable declarations:
	LOGICAL its_dynamic
	INTEGER NCF_GET_AGG_COUNT, TM_LENSTR1,
     .		cx_orig, agg_cx, hi_cx, big_cx, isp_orig, agg_dset, 
     .		agg_len, imemb, fer_uvar, memb_dset, memb_grid, iline, 
     .		nv, test_grid, this_grid, len_name, agg_dim, gnum, var

	CHARACTER name*512

#include "netcdf.inc"
	include	'tmap_errors.parm'
	include 'tmap_dims.parm'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'
	include	'xfr_grid.cmn'
	include	'xdset_parms.cmn'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xdset_info.cd_equiv'

*        ... interpretation stack setup at the time this is called ...
*	stack level		stack contents
*	isp+1	<--	level to get grids for components
*	 isp	<--	current level: evaluate user-variable grid

* --- end of introductory code ---

* initialize

	agg_dim = e_dim
	test_grid = unspecified_int4

	agg_dset = dset
	status = NCF_GET_AGG_COUNT (agg_dset, agg_len)

	agg_cx = cx

	name = uvar_text(uvar)
	len_name = TM_LENSTR1(name)

	DO 90 imemb = 1, agg_len

* See if the previous two members have matching grids (except for ensemble direction)
	   IF (test_grid.NE.unspecified_int4 .AND. test_grid.NE.this_grid) THEN
	      DO 80 idim = 1, nferdims
	         IF ( (idim .NE. agg_dim) .AND. 
     .	              (grid_line(idim,this_grid) .NE. grid_line(idim,test_grid)) ) GOTO 5300

   80	      CONTINUE
	   ENDIF

	   CALL CD_GET_AGG_VAR_INFO (agg_dset, name, imemb, 
     .         fer_uvar, memb_dset, memb_grid, iline, nv, status)
	   IF (fer_uvar .EQ. cat_file_var) THEN
	      IF (memb_grid .NE. unspecified_int4) THEN
	         test_grid = memb_grid
		 this_grid = memb_grid
		 GOTO 90
	      ENDIF
	   ENDIF

* Get ready to call is_uvar_grid. 
* pre-load the variable info onto the CX stack

	   cx_orig = cx_stack_ptr
	   CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   lo_cx  = is_cx( isp )
	   hi_cx = cx_stack_ptr
	   CALL GET_VAR_CONTEXT (  name(:len_name),
     .				lo_cx,
     .				hi_cx,
     .				status )

	   cx_data_set(hi_cx) = memb_dset
	   cx_grid(hi_cx) = memb_grid
	   cx_variable(hi_cx) = nv
	   cx_category(hi_cx) = cat_user_var

* pre-load the variable info onto the IS stack
	   isp_orig = isp
	   CALL PUSH_INTERP_STACK( isp, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   is_cx   ( isp ) = hi_cx
	   is_uvar ( isp ) = nv
	   is_act  ( isp ) = isact_get_grid

* initialize the stacks - (repeated for each component)

 100	   CALL IS_GRID_MODE( status )
           hi_cx = is_big_cx(isp) + 1  ! ???
	   IF ( status .NE. ferr_ok ) GOTO 5000

* evaluate the grid by merging component grids
*  ... exit 100 is to evaluate the grid for a component uvar
*  ... exit 300 means that a grid has been obtained
*  ... "normal" return indicates an error has occurred
 200	   CALL IS_UVAR_GRID( *100, *300, status )
	   GOTO 5000

* a grid has been obtained.  Are we done ?
 300	   IF ( isp .EQ. isp_orig ) THEN
* ... yes - clean up and go home
              this_grid = cx_grid(hi_cx) 
	      IF ( cx_stack_ptr .NE. cx_orig ) WRITE (6,*) 'agg_v_gd'	! temp
	      GOTO 90
	   ELSE
* ... no - go back for more
	      GOTO 200
	   ENDIF

 90	CONTINUE

* Test that the grids match for the last aggregate member (except for ensemble direction)

	IF (test_grid.NE.unspecified_int4 .AND. test_grid.NE.this_grid) THEN
	   DO 180 idim = 1, nferdims
	      IF ( (idim .NE. agg_dim) .AND. 
     .	           (grid_line(idim,this_grid) .NE. grid_line(idim,test_grid)) ) GOTO 5300
 180	CONTINUE
	ENDIF

* Define the aggregrate grid here. Need a new context?  Should this be done when
* we get back to is_uvar_grid?

	big_cx = is_big_cx( isp )
	
	igrid = cx_grid(big_cx)
	DO 400 idim = 1, nferdims
	   grid_line(idim, igrid) = grid_line(idim,test_grid)
  400	CONTINUE
	grid_line(agg_dim, igrid) = iline

* catalog this result checking for matching previously existing grid
	CALL TM_GET_LIKE_DYN_GRID(igrid, its_dynamic, gnum, 
     .				grd_stk_ptr, status ) 
	IF ( its_dynamic ) THEN
	   IF ( mode_diagnostic )  CALL DIAG_OP
     .           ('allocate', isact_class_impl_grid, gnum, 0)
	ENDIF

* set context for aggregate grid
	cx_grid(agg_cx) = gnum
	uvar_grid( uvar, dset ) = gnum 
	ds_grid_number(var) = gnum 

	cx_grid(agg_cx) = gnum
	cx_data_set(agg_cx) = agg_dset
	cx_category(agg_cx) = cat_aggregate_var

	uvar_grid( uvar, dset ) = gnum 
	uvar_dset( uvar ) = dset
	uvar_need_dset( uvar ) = .TRUE.

	var = cx_variable(agg_cx)
	ds_grid_number(var) = gnum 

* Get ds_grid_start, ds_grid_end, ds_ordering from the last file variable
	DO 730 idim = 1, nferdims
	   ds_grid_start(idim,var)= ds_grid_start(idim,nv)
	   ds_grid_end(idim,var)  = ds_grid_end(idim,nv)
	   ds_ordering(idim,var)  = ds_ordering(idim,nv)
 730	CONTINUE
	ds_ordering(agg_dim,var) = agg_dim
	ds_grid_start(agg_dim,var) = 1
	ds_grid_end(agg_dim,var) = line_dim(iline)

	is_cx(isp) = agg_cx
	is_uvar ( isp ) = uvar

* Now we have the grid for the variable in the agg. dataset. Store the grid 
* number for future use.
	DO 800 imemb = 1, agg_len
	   CALL CD_PUT_AGG_MEMB_GRID (agg_dset, name, imemb, gnum, status)
 800	CONTINUE

	RETURN

* error exit
 5000	CALL DEALLO_ALL_GRIDS
	RETURN
 5300	CALL ERRMSG( ferr_aggregate_error, status,
     .   'grids for LET/D variables do not match for aggregate dataset'
     .   //name,*5000 )

	END	
