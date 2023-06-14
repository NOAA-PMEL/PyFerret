        SUBROUTINE RIBBON_MISSING

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

*       Sets the options for missing-data representation in RIBBON plots
*       (see shade_set for color spectra)
* PyFr *kms* 5/13 Add alpha channel (opaque)

        IMPLICIT NONE
        include         'shade_vars.cmn'
        include 	'cmrdl_inc.decl'            
        include 	'CMRDL.INC'            
        include 	'gkscm1_inc.decl'            
        include 	'GKSCM1.INC'            
        include 	'gkscm2.cmn'            

	include        'gkspar.inc'


       	logical		there
        integer         beg,end                            
        character*2048   str                                

******************************************************************************

      	call upnsquish (label,str,end)
        call squish (label,1,2048)

*	If a spectrum is to be read, do that and set color
	beg = index (str,'SPECTRUM')
	if (beg .ne. 0) then
	   call RIBBON_READ (label,beg,end,there)
	   if (.not.there) return
	end if

*	Otherwise: if default colors requested, set them
	beg = index (str,'DEFAULT')
	if (beg .ne. 0) then
	   ribbon_miss(1) = -1.
	   ribbon_miss(2) = -1.
	   ribbon_miss(3) = -1.
	   ribbon_miss(4) = -1.
	end if

*	Otherwise: if user has said BLANK, mark this with -2.
	beg = index (str,'BLANK')
	if (beg .ne. 0) then
	   ribbon_miss(1) = -2.
	   ribbon_miss(2) = -2.
	   ribbon_miss(3) = -2.
	   ribbon_miss(4) = -2.
	end if

	return
	end


	SUBROUTINE RIBBON_READ (str,beg,end,there)

*	Mod *jd* 10.92 to allow memory reads
*	Mod *jd* 3.1.93 to use tm_inq_path

        include 	'tmap_errors.parm'    
	include		'cmrd_inc.decl'
	include		'CMRD.INC'
        include         'shade_vars.cmn'

	logical		there
	integer		beg,end,i,j,speclen,lnblk,status
        integer         test_end
	character*2048	str,sfile,instr,label,specdir,tm_inq_path
        character*2048   test_str
	external 	lnblk

        integer         mapping,percentage,value,level 
        INTEGER         TM_LENSTR1, ilen, slen
*******************************************************************************

*       DEFAULT MAPPING IS PERCENT FOR COMPATIBILITY WITH ORIGINAL SYSTEM
        rgb_mapping = percent

*	If no file name given, or its default, the line is black.
	if (end .lt. 10) then
	   there = .true.
	   return
	endif

	sfile = tm_inq_path (str(beg+9:end),'FER_PALETTE','.spk',
     .	.false.,status)

	if (status .ne. merr_ok) go to 100
	there = .true.

*	OPEN THE FILE

        open (70,file=sfile,status='old',err=100)

	rgb_mapping = percent  ! always: 

*	NOW READ IN THE FIRST LEVEL AND SET THE RGB 
10	read (70,'(a)',end=30,err=90) instr
	do 15 i = 1,80
	   if (instr(i:i) .ne. ' ') then
	      j = i
	      goto 20
	   end if
15	end do
	goto 10

*	PERMIT COMMENTS 
20	if (instr(j:j) .eq. '!') goto 10

*       IGNORE MAPPING
        if (index(test_str, 'RGB_MAPPING') .ne. 0) goto 10

* read first color, set color by percent.
	call upnsquish (instr,test_str,test_end)
 
        call parsex (instr(j:j+79),80,p,iflg,m,label,n)
	
* Leave the values as percentages. Will be calling SET_ONE_COLOR
* which expects this.

	if (p(1) .GE. 0. .and. p(1) .LE. 100.) THEN
	   ribbon_miss(1) = p(2)
	   ribbon_miss(2) = p(3)
	   ribbon_miss(3) = p(4)
           if ( m .eq. 5 ) then
              ribbon_miss(4) = p(5)
           else
              ribbon_miss(4) = 100.0
           endif
	ENDIF
	goto 30

30	close (70)
	return

*	ERROR WITH FILE ACCESS
90	close (70)
100	continue
	CALL WARN ('Unable to read ' //str(beg+9:end)//
     .	'.spk" to set missing-color for ribbon plot.')
	CALL WARN ('Defaulting to thin black line')

	return

	end

