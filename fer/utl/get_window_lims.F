	SUBROUTINE GET_WINDOW_LIMS( src_lo, src_hi, src_grid,
     .				 dst_lo, dst_hi, dst_grid,
     .				 idim, sslo, sshi )

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
	INTEGER sslo(dst_lo:dst_hi+1),sshi(dst_lo:dst_hi+1)

* internal variable declarations:
	INTEGER	ISUBSCRIPT, ii, dst_ss, status
	REAL*8	TDEST_WORLD, src_ww

* initialize
        IF ( idim .EQ. t_dim ) THEN
*         TDST_WORLD gives dates from the destination axis as they would be
*         encoded on the source axis 
*   ==>   ALL TIME/DATE CALCULATIONS ARE DONE IN THE DESTINATION AXIS ENCODING
           CALL TDEST_WORLD_INIT ( src_grid, dst_grid, status ) ! src<->dst
           IF ( status .NE. ferr_ok ) RETURN   ! note - status not passed!!!
        ENDIF

* pre-set the output to reflect no source points in any destination cell
	DO 10 ii = dst_lo, dst_hi
	   sslo(ii) = src_hi+1
	   sshi(ii) = src_lo-1
 10	CONTINUE

* loop through the source points identifying which cell each belongs in
	DO 100 ii = src_lo, src_hi

* ... find the destination cell containing the source point
	   src_ww = TDEST_WORLD( ii, src_grid, idim, box_middle )
	   dst_ss = ISUBSCRIPT( src_ww, dst_grid, idim, round_up)

	   IF ( dst_lo.LE.dst_ss .AND. dst_ss.LE.dst_hi ) THEN
	      sslo(dst_ss) = MIN(sslo(dst_ss), ii)
	      sshi(dst_ss) = ii   ! MAX() not needed -> ii strictly increasing
	   ENDIF

 100	CONTINUE

	RETURN
	END	
