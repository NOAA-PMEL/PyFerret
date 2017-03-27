	SUBROUTINE PURGE_ALL_UVARS_EXCEPT_EX

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
* eliminate all stored user variables from memory except for the current EX#
* expression

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V702 *sh* 1/17 - for dynamic memory -- based upon PURGE_ALL_UVARS

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xvariables.cmn'

* internal variable declarations:
	INTEGER	uvar, i, j, num_indices

        CALL deleted_list_get_undel(uvar_num_items_head, 
     .                          deleted_list_result(1),
     .                          max_uvar, num_indices )
        DO j = 1, num_indices
           uvar = deleted_list_result(j)

* do not delete EX#n variables as they may be actively in use
* as in "set att temp.missing_value = 12"
	   IF (uvar_name_code(uvar)(1:3) .EQ. 'EX#'    ) CYCLE

* eliminate all memory resident instances of this variable
	   CALL PURGE_USER_VAR( uvar )

* back to initial defaults (see INIT_UVAR)
* (eliminate all grid information that may have been rendered incorrect)
	   CALL DEALLO_UVAR_GRIDS(uvar)

*       bug fix: 11/91 *************
* must reset ALL of the state variables as set up in INIT_UVAR
* by default assume the definition requires an external data set specification
           uvar_need_dset( uvar ) = .TRUE.

* by default assume that all limits are irrelevant
           DO 320 i = 1, nferdims
 320       uvar_given(i,uvar) = uvlim_irrelevant
* end of bug fix: 11/91 *************

	ENDDO

	CALL PURGE_AUX_VAR_DEPENDENTS

	RETURN
	END
