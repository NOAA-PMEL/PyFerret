	SUBROUTINE EXTRACT_LINE( cx, src, msrc, line, idim, ndim, n, status )

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
* extract a single line of data from a memory variable grid.  Determine
* which line to extract from the given context

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/8/87
* V200:  5/19/89 - 4D symmetrical
*	10/16/89 - modified array declarations using call to *_SUB
*	  5/9/90 - allow a multi-dimensional region as a line for PLOT/VS
*	         - return number of dims

#ifdef unix
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	cx, msrc, idim, ndim, n, status
	REAL    src(*), line(*)

* internal variable declarations:
	INTEGER	CGRID_SIZE, dim(4)

* determine 4D shape of context
	CALL GET_CX_DIMS( cx, ndim, dim )

* have we been given a reasonable context ?
! eliminated 5/90
!	IF ( ndim .NE. 1 ) CALL ERRMSG
!     .		( ferr_dim_underspec, status, 'can only PLOT lines', *5000 )

* determine the axis orientation and length
	idim = dim(1)
	n    = CGRID_SIZE( cx )

* extract the line
	CALL EXTRACT_LINE_SUB(	cx, line, src,
     .				mr_lo_s1(msrc), mr_hi_s1(msrc),
     .				mr_lo_s2(msrc), mr_hi_s2(msrc),
     .				mr_lo_s3(msrc), mr_hi_s3(msrc),
     .				mr_lo_s4(msrc), mr_hi_s4(msrc)  )

* successful completion
	status = ferr_ok
	RETURN

! 5000	RETURN
	END
