        SUBROUTINE MOVIE_FRAME_RIS8 (ws_id,frm_status,ris_file,
     .  do_palette,rle,xdim,ydim,scan_line,pixmap)

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

