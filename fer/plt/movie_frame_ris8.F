        SUBROUTINE MOVIE_FRAME_RIS8 (ws_id,frm_status,ris_file,
     .  do_palette,rle,xdim,ydim,scan_line,pixmap)

*       Modification of original MOVIE_FRAME to capture GKS graphical output
*       in a pixmap.  This routine stores images in HDF RIS8 format for
*       playback by X Data Slice (XDS) or Ximage

*       J Davison 12.23.92
* Linux Port *kob* 1/97 - added check for type byte, because fortran 90 
*                         does not accept this type.

#ifdef unix
        include 'ferret_cmn/errmsg.parm'
#else
        INCLUDE 'FERRET_CMN:ERRMSG.PARM'
#endif

	integer		error_status,ws_id
	integer		xdim,ydim,x_pixel_dim,y_pixel_dim
	integer	        frm_status,rle
	integer	        scan_line(xdim,1)
#ifdef NEED_BYTE
	byte            pixmap(xdim,ydim)
#else
        integer*1       pixmap(xdim,ydim)
#endif
	logical		do_palette
        character*80    ris_file

******************************************************************************

*	CHECK THAT FRAME SIZE IS AS EXPECTED
	call get_frame_size (ws_id,error_status,x_pixel_dim,y_pixel_dim)
	if (error_status .ne. ferr_ok .or.
     .	    xdim .ne. x_pixel_dim .or. ydim .ne. y_pixel_dim) goto 1000

*	STORE THE PRESENT COLOR MAP?
	if (do_palette) call store_palette (ws_id)

*	CALL SNAPIT TO CAPTURE THE PIXMAP
	call snapit_ris8 (ws_id,rle,xdim,ydim,error_status,ris_file,
     .	scan_line,pixmap)

	if (error_status .ne. 0) goto 1000
        frm_status = ferr_ok
        return

*	GKS ERROR EXIT
1000	frm_status = ferr_tmap_error
	return

	end

