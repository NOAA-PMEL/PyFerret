	SUBROUTINE GET_WINDOW_LIMS( src_lo, src_hi, src_grid,
     .				 dst_lo, dst_hi, dst_grid,
     .				 idim, subsc )

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
* for regridding axis idim from source grid to destination grid:
* determine source grid points contained within the destination boxes

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V510 - *sh* 1/00 -- drawn from GET_AVE_LIMS

#ifdef unix
	include 'ferret.parm'
	include 'errmsg.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
#endif

* calling argument declarations:
	INTEGER src_lo, src_hi, dst_lo, dst_hi, src_grid, dst_grid, idim
	INTEGER subsc(dst_lo:dst_hi+1)

* internal variable declarations:
	INTEGER	ISUBSCRIPT, ii, src_ss, status
	REAL*8	TDEST_WORLD, xdst

* initialize
        IF ( idim .EQ. t_dim ) THEN
*         TDST_WORLD gives dates from the source axis as they would be
*         encoded on the destination axis 
*   ==>   ALL TIME/DATE CALCULATIONS ARE DONE IN THE SOURCE AXIS ENCODING
           CALL TDEST_WORLD_INIT ( dst_grid, src_grid, status ) ! src<->dst
           IF ( status .NE. ferr_ok ) RETURN   ! note - status not passed!!!
        ENDIF

* find lower bounds of destination grid boxes on source axis
	DO 100 ii = dst_lo, dst_hi

	   xdst    = TDEST_WORLD( ii, dst_grid, idim, box_lo_lim )
	   src_ss  = ISUBSCRIPT( xdst, src_grid, idim, round_up)

	   IF ( src_ss.LT.src_lo .OR. src_ss.GT.src_hi ) THEN
	      subsc(ii) = unspecified_int4
	   ELSE
	      subsc(ii) = src_ss
	   ENDIF

 100	CONTINUE

* top dest box always searches all the way to the top of the source list
	subsc( dst_hi+1 ) = src_hi

	RETURN
	END	
