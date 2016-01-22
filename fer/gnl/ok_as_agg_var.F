	LOGICAL FUNCTION OK_AS_AGG_VAR( var1, cat1, dset1, memb_list,
     .		             nsets, agg_dim, quiet, ref_fvar, status )
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

* determine if a candidate variable from the fist datset of an aggregation
* is found and has a suitable grid in all other members of the aggregation
* make sure the grid of all member variables is known -- even LET/D vars
* at least one member must have a file variable of this name -- the ref_fvar

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - TMAP 
*
* V698 12/15 *sh* extracted from init_ef_aggregate_dset.F

* arguments:
* inputs:
*	var1	  - variable number of the candidate var from dset 1
*	cat1	  - category (uvar or file) of the candidate var from dset 1
*	dset1	  - the dataset number of the 1st member
*	memb_list - list of member datasets
*	nsets	  - number of datasets in the aggregation
*	agg_dim	  - axis of aggregation
*	quiet	  - flag whether to announce unacceptable variables
* outputs
*	ref_fvar  - the first member file variable with an acceptable grid
*	status	  - return status

      IMPLICIT NONE
	include 'netcdf.inc'
#include "tmap_dset.parm"
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'xdset_info.cmn_text'
	include 'xdset_info.cd_equiv'
	include 'xunits.cmn_text'
	include 'xtm_grid.cmn_text'  
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xfr_grid.cmn'

* calling argument declarations:
	LOGICAL quiet
	INTEGER	var1, cat1, dset1, nsets, memb_list(nsets), agg_dim,
     .		ref_fvar,status

* local variable declarations:
	LOGICAL found_name
	INTEGER iset, member, j, iv, var, cat, grid, ref_grid, idim, 
     .		vlen, nuvars
	CHARACTER vname*128
	INTEGER	STR_SAME, TM_LENSTR1, KNOWN_GRID

* initialize
	status = ferr_ok
	ref_fvar = unspecified_int4

* the name of the candidate variable
	IF (cat1 .EQ. cat_file_var) THEN
	   vname = ds_var_code(var1)
	ELSE
	   vname = uvar_name_code(var1)
	ENDIF 
	vlen = TM_LENSTR1(vname)

* is the candidate variable suitable as the reference fvar?
	IF (cat1 .EQ. cat_file_var) THEN
	   grid = ds_grid_number(var1)
	   IF (grid_line(agg_dim, grid) .NE. mnormal) GOTO 5400
	   IF ( (agg_dim.EQ.f_dim .AND. grid_line(t_dim,grid).NE.mnormal) 
     .	  .OR.  (agg_dim.EQ.e_dim) )          THEN
	      ref_fvar = var1
	      ref_grid = ds_grid_number(ref_fvar)
	   ENDIF
	ELSE
* ... no but make sure we know its grid
	   CALL GET_UVAR_GRID( var1, dset1, status )
           IF ( status .NE. ferr_ok ) GOTO 5500
           grid = KNOWN_GRID(dset1,cat_user_var,var1)
	ENDIF 

* *** loop over the member datasets ***
	DO iset = 2, nsets
	   member = memb_list(iset)
	   IF (member .EQ. pdset_dummy) CYCLE ! ignore dummy (missing) dataset
	   found_name = .FALSE.

	   CALL FIND_VAR_NAME (member,vname(:vlen), cat, var )
	   IF (var .EQ. munknown_var_name) GOTO 5100
	   IF (cat .EQ. cat_user_var) THEN
	      IF (uvar_dset(var).EQ. unspecified_int4) THEN
	         GOTO 5600
	      ELSEIF (uvar_dset(var).NE. member) THEN
	         GOTO 5100
	      ENDIF
* make sure the grid of this member LET/D variable is known
	      CALL GET_UVAR_GRID( var, member, status )
              IF ( status .NE. ferr_ok ) GOTO 5500
              grid = KNOWN_GRID(member,cat_user_var,var)
	   ENDIF


* "reference fvar" -- the first file var encountered with an acceptable grid
	   IF ( ref_fvar.EQ.unspecified_int4
     .	  .AND. cat .EQ. cat_file_var          ) THEN
	      grid = ds_grid_number(var)
	      IF (grid_line(agg_dim, grid) .NE. mnormal) GOTO 5400
	      IF ( (agg_dim.EQ.f_dim .AND. grid_line(t_dim,grid).NE.mnormal) 
     .	     .OR.  (agg_dim.EQ.e_dim) )          THEN
	         ref_fvar = var
	         ref_grid = ds_grid_number(ref_fvar)
	      ENDIF
	   ENDIF

* now check grid compatibility
	   IF (ref_fvar .EQ. unspecified_int4) THEN
	      GOTO 5000
	   ELSE
	      IF (cat .EQ. cat_file_var) THEN
	         grid = ds_grid_number(var)
	      ELSE
	         grid = uvar_grid(var, member)
	      ENDIF
* ... check that the grid of this variable conforms with the reference grid
	      IF (grid_line(agg_dim, grid) .NE. mnormal) GOTO 5400
	      DO idim = 1, nferdims
	         IF (agg_dim.EQ.f_dim .AND. idim.EQ.t_dim) CYCLE
		 IF ( grid_line(idim, ref_grid) .NE.  
     .                grid_line(idim, grid)         ) GOTO 5200
	      ENDDO
	   ENDIF
	ENDDO

* success
	OK_AS_AGG_VAR = .TRUE.
	RETURN

* failure conditions
 5000	OK_AS_AGG_VAR = .FALSE.
	RETURN
 5100	IF ( .NOT.quiet ) 
     .	         CALL WARN('Exclude variable from aggregate. '//
     .	         'Does not appear in all member datasets: '//
     .	         vname(:vlen) )
	GOTO 5000

 5200	IF ( .NOT.quiet ) 
     .	         CALL WARN('Exclude variable from aggregate. '//
     .	         'Grid differs in member datasets: '//
     .	         vname(:vlen) )
	GOTO 5000

 5400	CALL ERRMSG( ferr_TMAP_error, status,
     .		' Variable in member dataset already '
     .		//'has axis in the aggregation direction: '//
     .		vname(:vlen), *5000 )

 5500   CALL ERRMSG( ferr_aggregate_error, status,
     .          'Could not get grid for user-defined variables '// 
     .		vname(:vlen), *5000 )

 5600   CALL ERRMSG( ferr_aggregate_error, status,
     .          'Conflicting global LET definition for '// 
     .		vname(:vlen), *5000 )


	END
