	SUBROUTINE GET_LINE_DYNMEM( rqst_size, iaxis, status )

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
* allocate the requested amount of dynamic coordinate storage (using c or python malloc)
* malloc is done in c (or python), but has to call back into F90 to set pointer
* the hierarchy of calls is as follows for allocating memory for coordinate storage.
*   get_ax_linemem.F ==> get_ax_mem.c ==> store_line_ptr.F

* programmer - Ansley Manke
* V72 6/17 *acm* For trac enhancement #767 -- dynamic coordinate storage 
*                Following main-memory dynamic allocation methods

                
        IMPLICIT NONE
	include	'tmap_dims.parm'
	include 'errmsg.parm'
	include	'ferret.parm'
#       include "tmap_dset.parm"
        include 'xtm_grid.cmn_text'
        external xgt_grid_data
	include 'xvariables.cmn'
	include 'xprog_state.cmn'

* calling argument declarations:
	INTEGER	iaxis, status
	INTEGER rqst_size

* internal variable declarations:

	INTEGER	num_slen
	INTEGER*8 i8_rqst_size
	CHARACTER LEFINT8*20, TM_FMT*12, num_str*20
	REAL	GET_LINE_COORD

	i8_rqst_size = rqst_size

	IF (line_allocated (iaxis) .GT. 0) THEN

	   IF (line_use_cnt(iaxis) .LE. 0) THEN
 	      IF ( iaxis .LE. max_lines ) THEN
 	         CALL FREE_LINE_DYNMEM ( iaxis )  ! checks that irreg coords are stored
		 line_name(iaxis) = char_init16
 	      ENDIF
 	   ENDIF

	ENDIF

* allocate the dynamic memory
        CALL GET_LINEMEM(iaxis, i8_rqst_size, status)
        IF (status .NE. ferr_ok) GOTO 5300

* and for edges
	i8_rqst_size = rqst_size + 1
        CALL GET_EDGMEM(iaxis, i8_rqst_size, status)
        IF (status .NE. ferr_ok) GOTO 5300

	line_allocated (iaxis) = rqst_size
	RETURN


***********************
* error exits
 5000	RETURN

 5300	CALL ERRMSG( ferr_insuff_memory,status,' ', *5310)
 5310	CALL SPLIT_LIST(pttmode_explct, show_lun,
     .		'    The OS refuses to supply memory for coordinate storage',0 )

        GOTO 5000

	END


