        SUBROUTINE GET_WINDOW_PIXELS (ws_id,x_pixel_dim,y_pixel_dim)

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

