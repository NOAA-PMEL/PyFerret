	SUBROUTINE SET_PAT_SIZE (wsid,xsize,ysize)

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
*	Sets the pattern size for patterned fill area calls -- needed
*	to keep sizes appropriate to the output device. 

*	XSIZE,YSIZE are x and y pattern size in pixels

*	J Davison JISAO/PMEL/TMAP
*	10.28.88

	real		window(4),vport(4),xres,yres
        integer         wsid,error,xform,cols,rows,xsize,ysize

********************************************************************************

*	FIND OUT THE DEVICE "RESOLUTION" IN WORLD COORDS/PIXEL
	call gqcntn (error,xform)
	call gqnt (xform,error,window,vport)

	call gqpxad (wsid,window(1),window(4),window(2),window(3),
     .	error,cols,rows)

	xres = abs((window(2) - window(1)))/real (cols)
	yres = abs((window(4) - window(3)))/real (rows)

*	SET PATTERN SIZE TO XSIZE BY YSIZE PIXELS
	call gspa (real (xsize)*xres,real (ysize)*yres)

	return
	end
