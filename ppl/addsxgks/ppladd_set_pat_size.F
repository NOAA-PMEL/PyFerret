	SUBROUTINE SET_PAT_SIZE (wsid,xsize,ysize)

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
