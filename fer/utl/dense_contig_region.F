	INTEGER FUNCTION DENSE_CONTIG_REGION(index_list, 
     .				istart, n,
     .				bad_flag, lo_cx_lim, hi_cx_lim )

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
* Given a *sorted* list of indices, select the largest slug (starting at i=1)
* that should be read as a contiguous region (a value between 1 and n)
* For example 1,2,3,5 could be read efficiently as 1:5 (only 20% "waste")
* but 1,5 should probably be read in 2 separate reads (60% "waste")

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V510 *sh* 8/99

#ifdef unix
	include 'errmsg.parm'
#else
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
#endif

* calling argument declarations:
	INTEGER n, istart, lo_cx_lim, hi_cx_lim
	REAL index_list(n), bad_flag

* internal variable declarations:
	INTEGER i

* interpal parameter declarations
	REAL useful_fraction
	PARAMETER (useful_fraction  = .6) !  1,3,5 ok, but 1,3,5,7 not


* is there anything left in the index buffer?
 50	IF (istart .GT. n) THEN
	   DENSE_CONTIG_REGION = 0
	   RETURN
	ENDIF

* work from the back forward to get the largest possible region
	DO 100 i = n, istart, -1
	  IF ( ((i-istart+1) / (index_list(i)-index_list(istart)+1))
     .		.GE. useful_fraction ) GOTO 1000
 100	CONTINUE

* done
* return the lo:hi limits of the region to be read
 1000	lo_cx_lim = index_list(istart)
	hi_cx_lim = index_list(i)

* return the pointer to the hi index of the region to be read
	DENSE_CONTIG_REGION = i
	RETURN

	END
