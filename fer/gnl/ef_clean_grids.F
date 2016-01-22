	SUBROUTINE EF_CLEAN_GRIDS( dset )


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
* check among the temporary grids for duplicates. Remove duplication.
* Ensure that distinct grids have distinct names.

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V698 1/16 *sh* based upon code from CD_CLEAN_GRIDS

* argument definitions:
*       dset    - pointer to aggregated data set

* include files
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
        include 'xbuild_grids.cmn'
	include 'xtm_grid.cmn_text'
	include 'xdset_info.cmn_text'
        include 'netcdf.inc'

* argument declarations
	INTEGER	 dset, status

* local variable declarations:
	LOGICAL TM_SAME_GRID_DEF
	INTEGER accepted_grid, test_grid, ivar
	CHARACTER buff*8	
                 
* check for duplicate grid guts
	accepted_grid = 0		! to get start of linked list
 10	CALL TM_NEXT_TMP_GRID( accepted_grid, *500)
	IF (grid_name(accepted_grid) .EQ. char_init) GOTO 10 ! already removed

	test_grid = accepted_grid
 20	CALL TM_NEXT_TMP_GRID( test_grid, *200)
	IF (grid_name(test_grid) .EQ. char_init) GOTO 100 ! already removed

* is the test grid a duplicate of the accepted grid?
	IF ( TM_SAME_GRID_DEF( accepted_grid, test_grid ) ) THEN
*  ... yes -- point its variables to the accepted grid
	   DO ivar = 1, maxvars
	      IF ( ds_var_setnum(ivar) .NE. dset ) CYCLE
	      IF ( ds_grid_number(ivar) .EQ. test_grid ) 
     .		   ds_grid_number(ivar) = accepted_grid
	   ENDDO
	   grid_name(test_grid) = char_init   ! throw it away
	ELSE
*  ... no -- make sure it has a unique name
	   CALL TM_NEW_GRID_NAME( grid_name(test_grid), buff )
	   grid_name(test_grid) = buff
	ENDIF
* inner loop -- Go back for another test_grid
 100	GOTO 20

* outer loop -- Go back for another accepted_grid
 200	GOTO 10

* we've tested the uniqueness of all temporary grids.  We're done.
 500	RETURN
	END
