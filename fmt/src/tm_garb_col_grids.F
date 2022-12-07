	SUBROUTINE TM_GARB_COL_GRIDS (dset)


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
* set use counts for grids and axes. Remove temporaries.

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92
* V5.0 *sh* 7/99 - with the change in definition to char_init this routine
*		may **for the first time** be doing work!!
*		Found a logic bug in the loop that re-assigns ds_grid_number
* V5.1 *sh* 4/00 - using dynamic grids and axes
* V674 2/12 *acm* 6D Ferret: use nferdims rather than 4 for dimension indexing
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax.

* argument definitions:
*	dset - the current data set

* include files
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'xtm_grid.cmn_text'
	        include 'xdset_info.cmn_text'
        external xdset_info_data

* calling argument declarations:
      INTEGER dset

* local variable declarations:
      INTEGER igrid, iaxis, idim

* reset the use counts of all newly-created grids to zero
* (They are set to 1 when allocated, but we set them to zero to put them
*  on a par with grids not created by this dtaa set, but used by it)
*  recatalog the grids defined by this data set
	igrid = 0		! to get start of linked list
 10	CALL TM_NEXT_TMP_GRID( igrid, *19)
	   grid_use_cnt(igrid) = 0
	   GOTO 10
 19	CONTINUE

* bump the use counts of grids used by variables in this data set
	CALL TM_DSET_USE_GRIDS( dset )

*  recatalog the grids defined by this data set
 100	igrid = 0		! to get start of linked list
	   CALL TM_NEXT_TMP_GRID( igrid, *199)

* ... bump use counts on all lines used by these grids
	   IF ( grid_name(igrid) .NE. char_init ) THEN
	      DO 110 idim = 1, nferdims
 	         CALL TM_USE_LINE( grid_line(idim,igrid) )
 110	      CONTINUE
!* ... a use count of zero indicates that this explicit parent grid is not used
!*	set its use count to 1 to indicate that it uses itself
!	      IF ( grid_use_cnt(igrid) .EQ. 0 ) grid_use_cnt(igrid) = 1

* ... recatalog as a valid dynamic grid
	      CALL TM_RE_ALLO_TMP_GRID( igrid )

	   ELSE
	      CALL TM_USE_DYN_GRID( igrid )  ! to avoid error trap
	      CALL TM_DEALLO_DYN_GRID_SUB( igrid )

	   ENDIF

	   GOTO 100
 199	CONTINUE


* recatalog the axes defined by this data set -- use counts are zero so far
 300	iaxis = 0	! to get start of linked list
	   CALL TM_NEXT_TMP_LINE( iaxis, *1000)

* ... add to list of valid dynamic lines
	   IF ( line_name(iaxis) .NE. char_init16 ) THEN

* ... a use count of zero indicates that this axis was defined but never used
*	(as, for example, in the file climatological_axes.cdf)
*	Flag it is a keeper -- requiring explicit deletion
	      IF ( line_use_cnt(iaxis) .EQ. 0 )
     .					line_keep_flag( iaxis ) = .TRUE.

	      CALL TM_RE_ALLO_TMP_LINE( iaxis )

	   ELSE
	      CALL TM_USE_LINE( iaxis )  ! to avoid error trap
	      CALL TM_DEALLO_DYN_LINE( iaxis )

	   ENDIF

	GOTO 300


* successful completion
 1000	RETURN

	END
