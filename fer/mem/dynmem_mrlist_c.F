	SUBROUTINE DYNMEM_MRLIST_C ( n, mrlist, c_ptr_array, status )

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
* pass the list of pointers implies by the list of pointers in mrlist
* into a c pointer list, where they will be readily accessible in c

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V702 1/17 *sh* 

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include	'xvariables.cmn'

* calling argument declarations:
	INTEGER	n, mrlist(n), status
        INTEGER*8 c_ptr_array

* internal variable declarations:
	INTEGER	i, mr

* create the c pointer array
	CALL DYNMEM_MAKE_PTR_ARRAY(n, c_ptr_array, status)
	IF (status .NE. ferr_ok) RETURN

* pass the FORTRAN pointers into c
	DO i = 1, n
	   mr = mrlist(i)
	   IF (mr .EQ. 0) THEN
	      CALL DYNMEM_PASS_1_PTR(i, memry(dummy_mr)%ptr, c_ptr_array)
	   ELSE
	      CALL DYNMEM_PASS_1_PTR(i, memry(mr)%ptr, c_ptr_array)
	   ENDIF
	ENDDO

	RETURN
	END

