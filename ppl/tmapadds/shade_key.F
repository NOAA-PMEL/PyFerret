        SUBROUTINE SHADE_KEY
        
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
*       Sets the options for SHADE key -- size,location,etc
*       J Davison JISAO/PMEL/TMAP 8.24.88
* v541 *acm*  5/02 add option to control outline on the shade key boxes
*                  also remove VMS INCLUDES

* 552 *acm* 4/03 Shade key improvements: Do not have to set all four corners 
*           when changing the size and location of the shade key -- can 
*           reset any of the four, and others will be filled in by making 
*           the key the default size in that direction.  See key.F

        include         'shade_vars.cmn'
	include		'cmrd_inc.decl'
	include		'CMRD.INC'
        include         'axis_inc.decl'
        include         'AXIS.INC'
        include         'plt_inc.decl'
        include         'PLT.INC'

        INTEGER i

*******************************************************************************


! From setup_defaults -- which is called only the first time from pplcmd_f

*      key defaults

c	do_key	     	= .true.   ! Still initialize this in setup_defaults.F
        vertical_key 	= .true.
        do 100 i = 1, 4
           kuser_loc(i) = .false.
  100   continue

	klab_size = 0.0
	klab_dig  = 3
	klab_len  = 9
	klab_incr = 0

	if (m .ge. 1 .and. iflg(1) .ne. 0 .and. p(1).eq.0.0) 
     .	do_key = .false.
	if (m .ge. 1 .and. iflg(1) .ne. 0 .and. p(1).eq.1.0) 
     .	do_key = .TRUE.

	if (m .ge. 2 .and. iflg(2) .ne. 0 .and. p(2).eq.0.0) 
     .	vertical_key = .false.
	if (m .ge. 2 .and. iflg(2) .ne. 0 .and. p(2).eq.1.0) 
     .	vertical_key = .TRUE.

	if (m .ge. 3 .and. iflg(3) .ne. 0) klab_size = p(3)
	if (m .ge. 4 .and. iflg(4) .ne. 0) klab_incr = ifix (p(4))
	if (m .ge. 5 .and. iflg(5) .ne. 0) klab_dig  = ifix (p(5))
	if (m .ge. 6 .and. iflg(6) .ne. 0) klab_len  = ifix (p(6))

* Set the user-indicated corners of the shade key. If a location is
* not marked, it will be set in key.F

        DO 1000 i = 1, 4
           kuser_loc(i) = (iflg(i+6) .NE. 0)           
 1000   CONTINUE

	IF (m .GE. 7) THEN

          IF (iflg(7) .NE. 0 .AND. iflg(8) .NE. 0 
     .	     .AND. iflg(9) .NE. 0 .AND. iflg(10) .NE. 0) THEN
	      kx_lo = p(7) * 1000.
	      kx_hi = p(8) * 1000.
	      ky_lo = p(9) * 1000.
	      ky_hi = p(10)* 1000.
           ELSE 

              IF (iflg(7).NE.0 .AND. iflg(8).EQ.0) kx_lo = p(7) * 1000.
              IF (iflg(7).EQ.0 .AND. iflg(8).NE.0) kx_hi = p(8) * 1000.
              IF (iflg(7).NE.0 .AND. iflg(8).NE.0) THEN
	         kx_lo = p(7) * 1000.
	         kx_hi = p(8) * 1000.
              ENDIF
  
              IF (iflg(9).NE.0 .AND. iflg(10).EQ.0) ky_lo = p(9) * 1000.
              IF (iflg(9).EQ.0 .AND. iflg(10).NE.0) ky_hi = p(10) * 1000.
              IF (iflg(9).NE.0 .AND. iflg(10).NE.0) THEN
	         ky_lo = p(9) * 1000.
	         ky_hi = p(10) * 1000.
              ENDIF
           ENDIF
        ENDIF

 2000   CONTINUE

        cont_key = .FALSE.
        IF (p(11) .eq. 1) cont_key = .TRUE.

        DO 3000 i = 1, 4
           iflg(i+6) = 0  ! Reset for subsequent plots
 3000   CONTINUE

	return
	end
