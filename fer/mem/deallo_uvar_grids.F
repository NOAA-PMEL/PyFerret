	SUBROUTINE DEALLO_UVAR_GRIDS( uvar )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* deallocate all dynamic grids associated with this user variable
* (clearing any memory references to this grid)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V420 9/8/95 - new
* V510 *sh* 4/00 - deallocate any grid, not just dynamic ones
* V533 *sh* 7/01 - set data type of grid/variable
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* V699 *sh* 6/16 - variable uvar_grids replaced with LISTs in NCF_Util.c 

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xvariables.cmn'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data

* calling argument declarations
	INTEGER	uvar

* local declarations
	INTEGER i, ngrids, nfound, saved_uvgrid, tmp_len
	CHARACTER tmp_buff*256

	nfound = 0
	CALL GET_SAVED_UVAR_GRID_LIST_LEN(uvar, ngrids)

* OLD CODE - based upon XVARIABLES uvar_grid variable
* loop through all of the grids that may be associated with this
* note: could be more efficient:  check to see if this is a data set
*	dependent variable (or "EX#n") and delete only the single data set grid

	DO 100 i = 0, max_gfdl_dsets

* Remove the grid
* (and possibly all memory resident variables that depend on it)
	   IF ( uvar_grid(uvar,i) .GE. 1
     .   .AND.  uvar_grid(uvar,i) .LE. max_dyn_grids ) THEN

* test the new LIST code
	      nfound = nfound + 1
	      CALL GET_SAVED_UVAR_GRID(uvar, i, saved_uvgrid)
	      IF (saved_uvgrid .NE. uvar_grid(uvar,i)) THEN
	         CALL GET_SHORT_DSET_NAME( i, tmp_buff, tmp_len)
	         CALL WARN("UVGRIDERR: deallo_uvar_grids")
	         CALL WARN("dset="//tmp_buff(:tmp_len))
	         WRITE(6,*) 'UVGRIDERR: deallo_uvar_grids'
	         WRITE(6,*)'dset='//tmp_buff(:tmp_len)
	      ENDIF

	      CALL DEALLO_DYN_GRID(uvar_grid(uvar,i))
	   ENDIF

* set the grid pointer to its initialized state
	uvar_grid( uvar, i ) = unspecified_int4

* set the data type of the grid/variable
	uvar_data_type( uvar, i ) = unspecified_int4

 100	CONTINUE


* free the LIST of grids associated with this uvar
	IF (nfound .NE. ngrids) THEN
	   CALL WARN("UVGRIDERR: deallo_uvar_grids nfound")
	   WRITE(6,*) 'UVGRIDERR: deallo_uvar_grids nfound'
	   WRITE(6,*)'uvar=', uvar
	ENDIF
	CALL FREE_UVAR_GRIDS(uvar)

	RETURN
	END
