	SUBROUTINE COPY_GRID ( src, msrc, dst, mdst )

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
* copy data from one grid to another (using limits of dstination grid)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 1/23/87
* V200:  5/19/89 - 4D symmetrical
*	10/11/89 - use COPY_GRID_SUB - avoid adjustable dimensioning with array
* V530:  8/00 *sh* - handle C string arrays, too
* V530:  8/00 *sh* - if input grid uses strides then copy based upon output
*	             grid limits -- the axes MUST BE the same length

	include 'ferret.parm'
	include 'errmsg.parm'
	include	'xvariables.cmn'

* calling argument declarations:
	INTEGER msrc, mdst
	REAL    src( * ), dst( * )

* internal variable declarations
	INTEGER	MGRID_SIZE
	INTEGER idim, lo_ss(4), hi_ss(4)

* set-up modified source limits wherever delta is significant
* cuz the FORTRAN dimensioned size can only have delta of 1 
	DO idim = 1,4
	  IF (mr_delta(idim,msrc) .NE. unspecified_val8) THEN
	    lo_ss(idim) = mr_lo_ss(mdst,idim)
	    hi_ss(idim) = mr_hi_ss(mdst,idim)
	  ELSE
	    lo_ss(idim) = mr_lo_ss(msrc,idim)
	    hi_ss(idim) = mr_hi_ss(msrc,idim)
	  ENDIF
	ENDDO

	IF ( mr_type(msrc) .EQ. ptype_string ) THEN

* set up the C pointer variable in the destination variable
	   CALL INIT_C_STRING_ARRAY( MGRID_SIZE(mdst),
     .				     dst,
     .				     mr_c_pointer(mdst)      )

* copy the strings
	   CALL COPY_PTR_GRID_SUB(
     .			  src, lo_ss(1), hi_ss(1),
     .			       lo_ss(2), hi_ss(2),
     .			       lo_ss(3), hi_ss(3),
     .			       lo_ss(4), hi_ss(4),
     .			  dst, mr_lo_s1(mdst), mr_hi_s1(mdst),
     .			       mr_lo_s2(mdst), mr_hi_s2(mdst),
     .			       mr_lo_s3(mdst), mr_hi_s3(mdst),
     .			       mr_lo_s4(mdst), mr_hi_s4(mdst) )

	ELSE

	   CALL COPY_GRID_SUB(
     .			  src, lo_ss(1), hi_ss(1),
     .			       lo_ss(2), hi_ss(2),
     .			       lo_ss(3), hi_ss(3),
     .			       lo_ss(4), hi_ss(4),
     .			       mr_bad_data( msrc ),
     .			  dst, mr_lo_s1(mdst), mr_hi_s1(mdst),
     .			       mr_lo_s2(mdst), mr_hi_s2(mdst),
     .			       mr_lo_s3(mdst), mr_hi_s3(mdst),
     .			       mr_lo_s4(mdst), mr_hi_s4(mdst),
     .			       mr_bad_data( mdst )			)

	ENDIF

 5000	RETURN
	END 
