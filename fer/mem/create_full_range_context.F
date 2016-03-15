	SUBROUTINE CREATE_FULL_RANGE_CONTEXT ( source, dest, status )

*
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
*
*
* transfer space/time/data_set/variable context from one context buffer to
* another and then expand it to be the full grid range of that variable

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V695+ 5/15 *sh*
* V698  3/16 *sh* - bug #2382: use CAXIS_LEN to get length of full axis

	IMPLICIT NONE

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xcontext.cmn'

* calling argument declarations
	INTEGER	source, dest, status

* internal variable declarations
	INTEGER CAXIS_LEN, idim, n

* initialize
	status = ferr_ok

	CALL TRANSFER_CONTEXT(source, dest)

* expand context to full grid range
	DO idim = 1, nferdims
	   cx_trans(idim,dest) = trans_no_transform
	   n = CAXIS_LEN(idim, dest)
	   IF ( n .GT. 0 ) THEN
	      cx_lo_ss(dest, idim) = 1
	      cx_hi_ss(dest, idim) = n
	      cx_by_ss(idim, dest) = .TRUE.
	      CALL FLESH_OUT_AXIS(idim, dest, status )
	      IF (status .NE. ferr_ok) GOTO 5000
	   ENDIF
	ENDDO

	RETURN

* error exit
 5000	RETURN
	END
