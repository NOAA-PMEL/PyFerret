        SUBROUTINE GET_WINDOW_PIXELS (ws_id,x_pixel_dim,y_pixel_dim)

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
*	Capture size in pixels of GKS ws with id ws_id
*       J Davison 12.23.92
*	Mod 3.11.93 for Ferret to capture entire screen when in viewport

*	Mod 10/9/95 *sh* to make this a PLOT+ routine rather than a Ferret
*	routine.  Renamed "GET_WINDOW_PIXELS" from "GET_FRAME_SIZE"

*	Note:  because this routine uses floating point calculations there
*	is reason for concern that there may be single pixel errors in the
*	size returned.  An alternative approach (in C) would use
*	XGKS gescinqxattr to get the X display, window, and graphic context
*	and (presumably) XGetWindowAttributes (see Ferret's grab_image_xwd.c)
*	to get the pixel size.
* 1/97 *kob* - Linux Port - had to move variable assignments off of
*               declaration line.

#ifdef unix
        include 'tmap_pplv11inc/pltcom_dat.decl'
        include 'pplv11inc/PLTCOM.DAT'
#else
        INCLUDE 'TMAP_PPLV11INC:PLTCOM_DAT.DECL'
        INCLUDE 'PPLV11INC:PLTCOM.DAT'
#endif

	integer		error_status,transno,ws_id
	integer		x_pixel_dim,y_pixel_dim
	integer		new_trans
	PARAMETER	( new_trans = 10 )

	real		fctx,fcty
	real		world(4),ndc(4),x_minw,x_maxw,y_minw,y_maxw

******************************************************************************

* initialize in case of error (not an adequate response, really)
	x_pixel_dim = 1
	y_pixel_dim = 1

*	GET WORLD COORDINATES AND PIXEL DIMENSIONS OF FRAME
	call  gqcntn (error_status,transno)
	if (error_status .ne. 0) RETURN

        IF(ASIZE.GT.BSIZE)THEN
           FCTY=BSIZE/ASIZE
           FCTX=1.0
        ELSE
           FCTX=ASIZE/BSIZE
           FCTY=1.0
        ENDIF
	CALL GSWN(new_trans, 0.0, asize, 0.0, bsize )
	CALL GSVP(new_trans, 0.0, fctx,  0.0, fcty )
	call gselnt (new_trans)

	call gqnt (new_trans,error_status,world,ndc)
	if (error_status .ne. 0) GOTO 1000

        x_minw = world(1)
        x_maxw = world(2)
        y_minw = world(3)
        y_maxw = world(4)

	call gqpxad(ws_id,x_minw,y_maxw,x_maxw,y_minw,
     .	error_status,x_pixel_dim,y_pixel_dim)

 1000	call gselnt (transno)
        return

	end

