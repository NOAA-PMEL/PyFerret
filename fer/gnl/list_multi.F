	SUBROUTINE LIST_MULTI( memory, mr_list, cx_list,
     .			       heading, perm, wsize, work, status )

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
* control routine to set up working arrays for LIST_MULTI_SUB

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  7/31/90
* V312: 5/94 - array "memory" as a calling argument

* calling argument declarations:
	LOGICAL	heading
	INTEGER	mr_list(*), cx_list(*), perm(4), wsize, status
	REAL	memory(*), work(0:wsize)
	
* internal variable declarations:
	INTEGER ws

* the work array is partitioned as follows in the called routine:
*	work(0)	     is xact_range(nvar)
*	work(ws)     is has_size(4,nvar)
*	work(5*ws)   is bad(nvar)
*	work(6*ws)   is pt_blk(nvar)
*	work(7*ws)   is size(4,nvar)
*	work(11*ws)  is msize(nvar)
*	work(12*ws)  is width(nvar)
*	work(13*ws)  is out(nvar)

	ws = wsize / 14
	CALL LIST_MULTI_SUB(memory,mr_list,cx_list,heading,perm, ws,
     .			work(0),     work(ws),   work(5*ws),
     .			work(6*ws),  work(7*ws), work(11*ws),
     .			work(12*ws), work(13*ws),
     .			status)


	RETURN
	END
