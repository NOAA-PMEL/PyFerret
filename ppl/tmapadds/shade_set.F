        SUBROUTINE SHADE_SET
        
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
*       J Davison JISAO/PMEL/TMAP 8.24.88
*	Completely rewritten J Davison 7.27.89
*       *JD* 4.22.91 Take out fill area rep setting for metafiles -- 
*				new translator
*	*jd* 3.93 Exchange DEFAULT and SPECTRUM code positions for FERRET
*	*jd* SPECTRA is now FER_PALETTE
*       *jd* eliminate call meta_protect for xgks -- but haven't replaced
*			functionality with other code
*	*jd* 6.94 add option to have comments in .spk files -- !-begun lines


#ifdef unix
        include         'tmap_pplv11inc/shade_vars.cmn'
        include 	'tmap_pplv11inc/cmrdl_inc.decl'            
        include 	'pplv11inc/CMRDL.INC'            
        include 	'tmap_pplv11inc/gkscm1_inc.decl'            
        include 	'pplv11inc/GKSCM1.INC'            
        include 	'tmap_pplv11inc/gkscm2.cmn'            
#else
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
        INCLUDE 	'tmap_pplv11inc:tmap_cmrdl.inc'            
        INCLUDE 	'tmap_pplv11inc:tmap_gkscm1.inc'            
        INCLUDE 	'tmap_pplv11inc:tmap_gkscm2.inc'            
#endif

#ifdef atc_gks
#   ifdef unix
	include        'atc_gksdir/gkspar.inc'
#   else
	INCLUDE        'atc_gksdir:gkspar.inc'
#   endif
#else
#   ifdef unix
  	include        'tmap_pplv11inc/gkspar.inc'
#   else
  	INCLUDE        'sys$library:gksdefs.bnd'
#   endif
#endif

       	logical		there
        integer         beg,end                            
        character*255   str                                

******************************************************************************

*	IF NO LABELS IN THE SHASET COMMAND THEN PROCESS LEVELS AS SPECIFIED
      	call upnsquish (label,str,end)
        call squish (label,1,255)
	if (end .eq. 0) then
	   call shade_levels
	   goto 10
	end if

*	IF A SPECTRUM IS TO BE READ, DO THAT AND SET LEVELS
	beg = index (str,'SPECTRUM')
	if (beg .ne. 0) then
	   call shade_read (label,beg,end,there)
	   if (.not.there) return
	   goto 10
	end if

*	OTHERWISE: IF DEFAULT COLORS REQUESTED, SET THEM
	beg = index (str,'DEFAULT')
	if (beg .ne. 0) then
	   call shade_default
	   goto 10
	end if

*	IF A SPECTRUM IS TO BE SAVED, DO THAT
	beg = index (str,'SAVE')
	if (beg .ne. 0) then
	   call shade_save (label,beg,end)
	   return
	end if

*	IF A SPECTRUM IS TO BE PROTECTED OR UNPROTECTED, DO THAT
	if (str .eq. 'PROTECT' .or. str .eq. 'RESET') then
	   call shade_protect_reset (str)
	   return
	end if

*       SET THE SHADES                                                
#ifdef atc_gks
10	call gsds (wsid,gasti,gallow) !This action necessary?
	if (.not. protect_used_colors) call set_fill_area_rep (wsid,wstype)
	call gsds (wsid,gasap,gsuppd)                                  
#else
 10	if (.not. protect_used_colors) 
     .	call set_fill_area_rep (wsid,wstype)
        call guwk (wsid,gpostp)
#endif
*4.91   if (meta_actv) call set_fill_area_rep (meta_wsid,meta_wstype)         

	return
	end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	SUBROUTINE SHADE_LEVELS 

#ifdef unix
        include         'tmap_pplv11inc/shade_vars.cmn'
	include		'tmap_pplv11inc/cmrd_inc.decl'
	include		'pplv11inc/CMRD.INC'
#else
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
	INCLUDE		'tmap_pplv11inc:tmap_cmrd.inc'
#endif

	integer		i,j

********************************************************************************

	custom_colors = .true.

*	NO PARAMETERS RESETS SPECTRUM TO GRAY SCALE
	if (m .eq. 0) then
	   ncl = 2

	   slvl(1) = 0.0
	   sr(1) = 0.0
	   sg(1) = 0.0
	   sb(1) = 0.0

	   slvl(2) = 100.0
	   sr(2) = 1.0
	   sg(2) = 1.0
	   sb(2) = 1.0

	   return
	end if

*	FIRST PARAMETER NEGATIVE DELETES THAT LEVEL
	if (p(1) .lt. 0.0 .and. p(1) .ne. -100.0) then
	   do 5 i = 1,ncl
	      if (p(1) .eq. -slvl(i))then
		 ncl = ncl - 1
		 do 1 j = i,ncl
		    slvl(j) = slvl(j+1)
		    sr(j) = sr(j+1)       
		    sg(j) = sg(j+1)       
		    sb(j) = sb(j+1)       
1		 end do
		 return
	      end if
5	   end do
	end if

*	NEED COMPLETE SET OF ARGUMENTS OTHERWISE
	if (m .ne. 4) return

*	REJECT PERCENT OF SPECTRUM OUTSIDE 0 TO 100	
	if (p(1) .lt. 0.0 .or. p(1) .gt. 100.0) return

*	REJECT PERCENT OF R,G,B OUTSIDE 0 TO 100
	if (p(2) .lt. 0.0 .or. p(2) .gt. 100.0) return
	if (p(3) .lt. 0.0 .or. p(3) .gt. 100.0) return
	if (p(4) .lt. 0.0 .or. p(4) .gt. 100.0) return

*	REPLACE OLDER VALUES WITH NEW IF RESPECIFIED
	do 10 i = 1,ncl
	   if (p(1) .eq. slvl(i)) then
	      sr(i) = p(2)/100.0
	      sg(i) = p(3)/100.0
	      sb(i) = p(4)/100.0

	      return
	   end if
10	end do

*	IF NEW, INSERT INTO PROPER LOCATION IN SPECTRUM
	do 20 i = 1,ncl-1
	   if (p(1) .lt. slvl(i+1)) then  
	      do 15 j = ncl,i+1,-1
		 slvl(j+1) = slvl(j)
		 sr(j+1) = sr(j)
		 sg(j+1) = sg(j)
		 sb(j+1) = sb(j)
15	      end do
	      ncl = ncl + 1

	      slvl(i+1) = p(1)
	      sr(i+1) = p(2)/100.0
	      sg(i+1) = p(3)/100.0
	      sb(i+1) = p(4)/100.0
	
	      return
	   end if
20	end do

	return
	end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	SUBROUTINE SHADE_DEFAULT

#ifdef unix
        include         'tmap_pplv11inc/shade_vars.cmn'
#else
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
#endif

********************************************************************************

	custom_colors = .false.
	return
	end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	SUBROUTINE SHADE_READ (str,beg,end,there)

*	Mod *jd* 10.92 to allow memory reads
*	Mod *jd* 3.1.93 to use tm_inq_path

#ifdef unix
        include 	'tmap_format/tmap_errors.parm'    
	include		'tmap_pplv11inc/cmrd_inc.decl'
	include		'pplv11inc/CMRD.INC'
        include         'tmap_pplv11inc/shade_vars.cmn'
#else

        INCLUDE 	'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE		'tmap_pplv11inc:tmap_cmrd.inc'
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
#endif

	logical		there
	integer		beg,end,i,j,speclen,lnblk,status
	character*255	str,sfile,instr,label,specdir,tm_inq_path

	external 	lnblk

*******************************************************************************

*	IF NO FILE NAME GIVEN, READ IN FROM MEMORY
	if (end .lt. 10) then
	   if (mncl .eq. 0) goto 100

	   custom_colors = .true.
	   ncl = mncl

	   do 4 i = 1,ncl
	      slvl(i) = mslvl(i)
	      sr(i) = msr(i)/100.0
	      sg(i) = msg(i)/100.0
	      sb(i) = msb(i)/100.0
4	   end do

	   there = .true.
	   return
	endif
************** Modified to use tm_inq_path *jd* 3.1.93 ***************
*#ifdef unix
*        call getenv ('SPECTRA',specdir)
*        speclen = lnblk (specdir,255)
*	sfile = specdir(:speclen)//'/'//str(beg+9:end)//'.spk'
*	inquire (file=sfile,exist=there)
*	if (there) goto 5

*	sfile = str(beg+9:end)//'.spk'
*	inquire (file=sfile,exist=there)
*	if (.not.there) goto 100
*#else
*	sfile = 'spectra:'//str(beg+9:end)//'.spk'
*	inquire (file=sfile,exist=there)
*	if (.not.there) goto 100
*#endif
***************** end mod 3.1.93 **************************************

	sfile = tm_inq_path (str(beg+9:end),'FER_PALETTE','.spk',
     .	.false.,status)
*	type *, sfile,str(beg+9:end),status,merr_ok
	if (status .ne. merr_ok) go to 100
	there = .true.

*	OPEN THE FILE
#ifdef atc_gks
 5      open (70,file=sfile,status='old',err=100)
#else
#ifdef NO_OPEN_READONLY
 5      open (70,file=sfile,status='old',err=100)
#else
 5      open (70,file=sfile,readonly,status='old',err=100)
#endif
#endif
*	RESET THE SPECTRUM FIRST OFF
	m = 0
	call shade_levels

*	NOW READ IN THE LEVELS AND SET THEM
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
	call parsex (instr(j:j+79),80,p,iflg,m,label,n)
	call shade_levels
	goto 10

30	close (70)
	return

*	ERROR WITH FILE ACCESS
90	close (70)
100	there = .false.
	write (6,'(/a/)') ' PPL+ error: Unable to read "' //
     .	str(beg+9:end)//'.spk" to set shade and fill colors.'

	return

	end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	SUBROUTINE SHADE_SAVE (str,beg,end)

#ifdef unix
        include         'tmap_pplv11inc/shade_vars.cmn'
#else
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
#endif

	integer		beg,end,i
	character*255	str,sfile

*******************************************************************************

*	WRITE TO MEMORY IF NO FILE NAME GIVEN
	if (end .lt. 6) then
	   mncl = ncl

	   do 10 i = 1,ncl
	      mslvl(i) = slvl(i)
	      msr(i) = sr(i)*100.0
 	      msg(i) = sg(i)*100.0
	      msb(i) = sb(i)*100.0
10	   end do
	   goto 101
	end if

#ifdef unix
	sfile = str(beg+5:end)//'.spk'

*	OPEN THE FILE

#   ifdef atc_gks
	open (70,file=sfile,status='unknown',form='formatted',
     .	err=100)
#   else
#ifdef NO_OPEN_CARRIAGECONTROL
	open (70,file=sfile,status='unknown',form='formatted',
     .	err=100)
#else
	open (70,file=sfile,status='unknown',form='formatted',
     .	carriagecontrol='list',err=100)
#endif
#   endif
#else
	sfile = 'fer_palette:'//str(beg+5:end)//'.spk'

*	OPEN THE FILE
#ifdef NO_OPEN_CARRIAGECONTROL
	open (70,file=sfile,status='new',form='formatted',
     .	err=100)
#else
	open (70,file=sfile,status='new',form='formatted',
     .	carriagecontrol='list',err=100)
#endif
#endif
*	WRITE OUT THE SPECTRUM
	do i = 1,ncl
	   write (70,'(4f7.1)') slvl(i),sr(i)*100.0,sg(i)*100.0,sb(i)*100.0
	end do

	close (70)
	return

100	write (6,'(/a/)') ' PPL+ error: Unable to save spectrum ' //
     .	str(beg+9:end)//'.spk"'
101	return
	end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	SUBROUTINE SHADE_PROTECT_RESET (str)

#ifdef unix
        include         'tmap_pplv11inc/shade_vars.cmn'
	include		'tmap_pplv11inc/gkscm1_inc.decl'
	include		'pplv11inc/GKSCM1.INC'
        include         'tmap_pplv11inc/gkscm2.cmn'

#else
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
        INCLUDE         'tmap_pplv11inc:tmap_gkscm1.inc'
        INCLUDE         'tmap_pplv11inc:tmap_gkscm2.inc'
#endif

	character*255	str

*******************************************************************************

*	PROTECT PREVIOUSLY USED COLORS
	if (str .eq. 'PROTECT') then
	   protect_used_colors = .true.
	elseif (str .eq. 'RESET') then
*	SET THEM TO BE REUSED
	   protect_used_colors = .false.
	   lo_color_ndx(wsid) = 0
	   hi_color_ndx(wsid) = 0
	endif

#ifndef xgks
	if (meta_actv .and. .not. meta_clsplt) call meta_protect
#endif

	return
	end
