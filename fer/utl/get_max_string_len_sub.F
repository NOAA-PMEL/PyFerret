	SUBROUTINE GET_MAX_STRING_LEN_SUB(cx, grid_data, 
     .		lox, hix, loy, hiy, loz, hiz, lot, hit, smax )

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
* Return the length of the longest string in this context

* V530: 8/00 *sh*
* v554: 12/04 *acm* Get the shape of the region for the context of the data;
*                   these are the indices to use in looping through grid_data.
*                   (fixes bug 617, listing string and numeric data together)

        include 'ferret.parm'
	include 'xcontext.cmn'

* calling argument declarations
	INTEGER  cx, smax, lox, hix, loy, hiy, loz, hiz,lot, hit
	REAL*8 grid_data( lox:hix,loy:hiy,loz:hiz,lot:hit )

* internal variable declarations:
	INTEGER  GET_C_STRING_LEN, i, j, k, l, slen, sslo(4), sshi(4),
     .         ndim, cxperm(4)

* initialize
	smax = 1

        CALL get_cx_dims (cx,ndim,cxperm)

        sslo(1) = cx_lo_s1(cx)
        sslo(2) = cx_lo_s2(cx)
        sslo(3) = cx_lo_s3(cx)
        sslo(4) = cx_lo_s4(cx)
        sshi(1) = cx_hi_s1(cx)
        sshi(2) = cx_hi_s2(cx)
        sshi(3) = cx_hi_s3(cx)
        sshi(4) = cx_hi_s4(cx)

	DO 100 l = sslo(cxperm(4)), sshi(cxperm(4))
	DO 100 k = sslo(cxperm(3)), sshi(cxperm(3))
	DO 100 j = sslo(cxperm(2)), sshi(cxperm(2))
	DO 100 i = sslo(cxperm(1)), sshi(cxperm(1))

	   slen = GET_C_STRING_LEN( grid_data(i,j,k,l) )
	   IF ( slen .GT. smax ) smax = slen
           
 100	CONTINUE

	RETURN
	END
 
