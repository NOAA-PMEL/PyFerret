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
* V699 *sh* 6/16 - variable uvar_grid replaced with LISTs in NCF_Util.c 
* V710 *acm* 18/25/16 - Ticket 2462 Issue the warnings only if in MODE DIAGNOSTIC 
* V710 *acm* 10/25/16 - Always issue the warnings: Memory-link fixes in NCF_Util.c 
* V710 *acm* 10/16 Ticket 2422: variables uvar_grid, uvar_data_type, uvar_aux_cat, 
*                  uvar_aux_var to be replaced with calls to LIST routines in NCF_Util.c 

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data

* calling argument declarations
	INTEGER	uvar

* local declarations
	INTEGER ngrids, saved_uvgrid, i, c_list_dset
	CHARACTER tmp_buff*256
	
	CALL GET_SAVED_UVAR_GRID_LIST_LEN(uvar, ngrids)

* THIS LOOP WILL BE REPLACED WITH a loop over ngrids (see above)
* using LIST controls that allow you to traverse the list from end-to-end:
* list_front, list_mvnext (?)

        IF (ngrids .GT. 0) THEN
	
* c_list_dset  - dataset value relevant in C LIST structures of NCF_Util.c

	   c_list_dset = uvar_dset(uvar)                    ! the dset that owns this LET/D
	   IF (c_list_dset .LE. pdset_irrelevant) c_list_dset = pdset_uvars  ! global uvars

	   DO i = 1, ngrids

	      CALL NCF_NEXT_UVAR_GRID_IN_LIST ( c_list_dset, uvar, i, saved_uvgrid )
	      IF ( saved_uvgrid.GE.1 .AND. saved_uvgrid.LE.max_dyn_grids ) 
     .           CALL DEALLO_DYN_GRID(saved_uvgrid)

	   ENDDO

        ENDIF

* free the LIST of grids associated with this uvar

	CALL FREE_UVAR_GRIDS(uvar)

	RETURN
	END
