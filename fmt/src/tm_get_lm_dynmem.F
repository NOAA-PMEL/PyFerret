	SUBROUTINE TM_GET_LM_DYNMEM( rqst_size, lm, status )

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
* allocate the requested amount of dynamic emory (using c or python malloc)
* malloc is done in c (or python), but has to call back into F90 to set pointer
* the hierarchy of calls is as follows
*   tm_get_lm_dynmem.F ==> tm_get_lm_mem.c ==> tm_store_lm_ptr.F

* programmer - steve hankin
* V702 *sh*  4/17 code copies/modified from fer/mem "mr" dynamic memory

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'tmap_errors.parm'
	include 'xdyn_linemem.cmn_text'

* calling argument declarations:
	INTEGER	lm, status
	INTEGER*8 rqst_size

* internal variable declarations:
	INTEGER	next, last
	CHARACTER*13    TM_STRING

* find a free table slot -- the first on the free list
	lm = free_dyn_line(plm_free_base)
	IF (lm .EQ. 0) GOTO 5100

* allocate the dynamic memory
	CALL TM_GET_LM_MEM(lm, rqst_size, status)
        IF (status .NE. merr_ok) GOTO 5000

* bookkeeping
        lm_size(lm) = rqst_size
        total_line_mem = total_line_mem + rqst_size
* ... remove this line from the free list
	free_dyn_line(plm_free_base) = free_dyn_line(lm)

	RETURN


***********************
* error exits
 5000	RETURN
 5100	CALL TM_ERRMSG (merr_linstorlim, status, 'TM_GET_LM_DYNMEM',
     .                  no_descfile, no_stepfile,
     .                  'MAX='//TM_STRING(DBLE(max_lm)),
     .                  no_errstring, *5000)

	END


