	SUBROUTINE TEK4014_LINE_BUNDLE (zwsid)

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
*	J Davison JISAO/PMEL/TMAP 3.23.89
*	*jd* 2.19.92 mod to support atc gks

	integer		zwsid,i,j
#ifdef atc_gks
        integer         ltype(6)
	data		ltype /1,2,3,4,-1001,-1004/
#else
	integer 	ltype(5)
	data		ltype /1,2,3,4,-3/
#endif
	real		thickness

*******************************************************************************

#ifdef atc_gks
*	INCREASE THICKNESS AND REPEAT LINETYPE EACH SIX INDICES
100     do 20 j = 1,3
	   thickness = real (j)*2.0 - 1.0

           do 10 i = 1,6
              call gsplr (zwsid,i+6*(j-1),ltype(i),thickness,1)
10        continue
20     continue
#else
*	INCREASE THICKNESS AND REPEAT LINETYPE EACH FIVE INDICES
	do 20 j = 1,3
	   thickness = real (j)*2.0 - 1.0

	   do 10 i = 1,5
	      call gsplr (zwsid,i+5*(j-1),ltype(i),thickness,1)
10	   continue
20	continue
#endif
	return

	end
