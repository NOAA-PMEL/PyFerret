        SUBROUTINE SHADE_SET

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

*       Sets the options for SHADE -- fill_missing, custom_colors
*       J Davison JISAO/PMEL/TMAP 8.24.88
*	Completely rewritten J Davison 7.27.89
*       *JD* 4.22.91 Take out fill area rep setting for metafiles -- 
*				new translator
*	*jd* 3.93 Exchange DEFAULT and SPECTRUM code positions for FERRET
*	*jd* SPECTRA is now FER_PALETTE
*       *jd* eliminate call meta_protect for xgks -- but haven't replaced
*			functionality with other code
*	*jd* 6.94 add option to have comments in .spk files -- !-begun lines
*       *jd* 10.98 Add control for rgb interpolation, atomic colors, or data
*                 based levels in color bar
*       *acm* 9/02 Check that the number of levels in a palette file isn't 
*                  larger than SPEC_SIZE (see shade_vars.cmn), the number 
*                  of spectrum levels allowed to be specified in a palette file.
* v552 *acm* 3/03 Longer strings: changes for multi-line labels.
* V602: *acm* 1/07 Changes for metafile-batch mode
* V608: *acm*11/07 the text for RGB_mapping must be palette, by_value, or by_level; issue
*                  a warning if its not one of these. Fixes bug 1549
* V610  *acm* 3/08 Fixes for metafile batch mode
* PyFr *kms* 5/13 Add alpha channel (opaque) variables SA,MSA

        IMPLICIT NONE
        include         'shade_vars.cmn'
        include 	'cmrdl_inc.decl'            
        include 	'CMRDL.INC'            
        include 	'gkscm1_inc.decl'            
        include 	'GKSCM1.INC'            
        include 	'gkscm2.cmn'            
	include         'gkspar.inc'

       	logical		there
        integer         beg,end,istate
        character*2048   str                                

******************************************************************************

      	call upnsquish (label,str,end)
        call squish (label,1,2048)

*	IF NO LABELS IN THE SHASET COMMAND THEN PROCESS LEVELS AS SPECIFIED
	if (end .eq. 0) then
	   call shade_levels
	   goto 10
	end if

*	IF RGB MAPPING MODIFIED, SET FLAG
	beg = index (str,'RGB_MAPPING')
	if (beg .ne. 0) then
	   call shade_mapping  (str)
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

10      CALL FGD_GQOPS(istate)  ! In batch mode the graphics may not be active now. 
        IF (istate .GE. 2) THEN

*       SET THE SHADES                                                
        IF ( .NOT. protect_used_colors )
     .     CALL set_fill_area_rep(wsid,wstype)

        ENDIF
	return
	end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	SUBROUTINE SHADE_LEVELS 

        IMPLICIT NONE
        include         'parampl5_dat.decl'
        include         'PARAMPL5.DAT'
        include         'cont_inc.decl'
        include         'CONT.INC'
        include         'shade_vars.cmn'
	include		'cmrd_inc.decl'
	include		'CMRD.INC'

	include 'errmsg.parm'

	integer		i,j, status

        CHARACTER*100   err_msg

*******************************************************************************

	custom_colors = .true.

*	NO PARAMETERS RESETS SPECTRUM TO GRAY SCALE
	if (m .eq. 0) then
           if (rgb_mapping .eq. by_value) then
              ncl = 0
           else
              ncl = 2
              if (rgb_mapping .eq. by_level) then
                 slvl(1) = 1.0
                 slvl(2) = 2.0
              else
                 slvl(1) = 0.0
                 slvl(2) = 100.0
              end if
           end if
           
           sr(1) = 0.0
           sg(1) = 0.0
           sb(1) = 0.0
           sa(1) = 1.0
           
           sr(2) = 1.0
           sg(2) = 1.0
           sb(2) = 1.0
           sa(2) = 1.0
           
           return
        end if
        
*       LISTING ONLY THE FIRST PARAMETER DELETES THAT LEVEL
	if (m .eq. 1) then
           if (rgb_mapping .eq. percent .and. 
     .          (p(1) .eq. 0.0 .or. abs(p(1)) .ge. 100.0)) return

           if (ncl .eq. 1) then
              ncl = 0
           else if (ncl .gt. 1) then
              do 5 i = 1,ncl
                 if (p(1) .eq. slvl(i)) then
                    ncl = ncl - 1
                    do 1 j = i,ncl
                       slvl(j) = slvl(j+1)
                       sr(j) = sr(j+1)       
                       sg(j) = sg(j+1)       
                       sb(j) = sb(j+1)       
                       sa(j) = sa(j+1)       
 1                  end do
                    return
                 end if
 5            end do
           end if
        end if

*	NEED COMPLETE SET OF ARGUMENTS OTHERWISE
	if ( (m .lt. 4) .or. (m .gt. 5) ) return

*	REJECT PERCENT OF SPECTRUM OUTSIDE 0 TO 100	
	if (rgb_mapping .eq. percent .and. 
     .       (p(1) .lt. 0.0 .or. p(1) .gt. 100.0)) return

*	REJECT PERCENT OF R,G,B,A OUTSIDE 0 TO 100
	if (p(2) .lt. 0.0 .or. p(2) .gt. 100.0) return
	if (p(3) .lt. 0.0 .or. p(3) .gt. 100.0) return
	if (p(4) .lt. 0.0 .or. p(4) .gt. 100.0) return
        if ( (m .eq. 5) .and. 
     .       (p(5) .lt. 0.0 .or. p(5) .gt. 100.0) ) return

*	IF FIRST VALUE, ENTER INTO TABLE
        if (ncl .eq. 0) then
           slvl(1) = p(1)
           sr(1) = p(2)/100.0
           sg(1) = p(3)/100.0
           sb(1) = p(4)/100.0
           if ( m .eq. 5 ) then
              sa(1) = p(5)/100.0
           else
              sa(1) = 1.0
           endif
           ncl = 1

           return
        end if
       
*	REPLACE OLDER VALUES WITH NEW IF RESPECIFIED
	do 10 i = 1,ncl
	   if (p(1) .eq. slvl(i)) then
	      sr(i) = p(2)/100.0
	      sg(i) = p(3)/100.0
	      sb(i) = p(4)/100.0
              if ( m .eq. 5 ) then
                 sa(i) = p(5)/100.0
              else
                 sa(i) = 1.0
              endif

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
                 sa(j+1) = sa(j)
15	      end do
	      ncl = ncl + 1
              IF (ncl .GT. spec_size) GOTO 1000

	      slvl(i+1) = p(1)
	      sr(i+1) = p(2)/100.0
	      sg(i+1) = p(3)/100.0
	      sb(i+1) = p(4)/100.0
              if ( m .eq. 5 ) then
                 sa(i+1) = p(5)/100.0
              else
                 sa(i+1) = 1.0
              endif

	      return
	   end if
20	end do
          
        if (p(1) .gt. slvl(ncl)) then 
           ncl = ncl + 1
           IF (ncl .GT. spec_size) GOTO 1000
           
           slvl(ncl) = p(1)
           sr(ncl) = p(2)/100.0
           sg(ncl) = p(3)/100.0
           sb(ncl) = p(4)/100.0
           if ( m .eq. 5 ) then
              sa(ncl) = p(5)/100.0
           else
              sa(ncl) = 1.0
           endif
        end if
 
 5000	RETURN
        
 1000	WRITE (err_msg, 110)  spec_size 
  110   FORMAT ('Too many spectrum levels in .spk file. Max is', i4)
        CALL ERRMSG( ferr_prog_limit, status,
     .		err_msg, *5000 )
	END

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	SUBROUTINE SHADE_DEFAULT

        IMPLICIT NONE
        include         'shade_vars.cmn'


*******************************************************************************

	custom_colors = .false.
	return
	end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	SUBROUTINE SHADE_MAPPING  (str)

* Set mapping for color bar:
*     PERCENT    -> 0
*     BY_VALUE   -> 1
*     BY_LEVEL   -> 2

        IMPLICIT NONE
        include         'shade_vars.cmn'

        character*2048   str

*******************************************************************************

	if (index (str, 'PERCENT') .ne. 0) then
           rgb_mapping = percent
           goto 10
	end if

        if (index (str, 'BY_VALUE') .ne. 0) then
           rgb_mapping = by_value
           goto 10
	end if

        if (index (str, 'BY_LEVEL') .ne. 0) then
           rgb_mapping = by_level
           goto 10
	end if

 10     return
	end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	SUBROUTINE SHADE_READ (str,beg,end,there)

*	Mod *jd* 10.92 to allow memory reads
*	Mod *jd* 3.1.93 to use tm_inq_path

        IMPLICIT NONE
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
        ncl = 0

*	IF NO FILE NAME GIVEN, READ IN FROM MEMORY
	if (end .lt. 10) then
	   if (mncl .eq. 0) goto 100

	   custom_colors = .true.
	   ncl = mncl
           rgb_mapping = mrgb_mapping

	   do 4 i = 1,ncl
	      slvl(i) = mslvl(i)
	      sr(i) = msr(i)/100.0
	      sg(i) = msg(i)/100.0
	      sb(i) = msb(i)/100.0
	      sa(i) = msa(i)/100.0
4	   end do

	   there = .true.
	   return
	endif
************** Modified to use tm_inq_path *jd* 3.1.93 ***************
*#ifdef unix
*        call getenv ('SPECTRA',specdir)
*        speclen = lnblk (specdir,2048)
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

*  CHECK IF NUMBER OF LEVELS IS LARGER THAN SPEC_SIZE, clr_ct=0

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

*       DETERMINE MAPPING
        call upnsquish (instr,test_str,test_end)
 
        if (index(test_str, 'RGB_MAPPING') .ne. 0) then
 
           percentage     = index (test_str, 'PERCENT')
           value          = index (test_str, 'BY_VALUE')
           level          = index (test_str, 'BY_LEVEL') 

           if (percentage .ne. 0) then
              rgb_mapping = percent
           else if (value .ne. 0) then
              rgb_mapping = by_value
           else if (level .ne. 0) then
              rgb_mapping = by_level
           else
              ilen = TM_LENSTR1(instr)
              slen = TM_LENSTR1(sfile)
              CALL WARN ('RGB_Mapping type not recognized, given '//
     .         'in palette file '//sfile(1:slen)//' as '//instr(1:ilen))
              CALL WARN ('Must be percent, by_value or by_level. '//
     .         'Defaulting to percent')
           endif
              

*	   RESET SPECTRUM TO GRAY SCALE
           if (rgb_mapping .eq. by_value) then
              ncl = 0
           else
              ncl = 2
              if (rgb_mapping .eq. by_level) then
                 slvl(1) = 1.0
                 slvl(2) = 2.0
              else
                 slvl(1) = 0.0
                 slvl(2) = 100.0
              end if
           end if
           
           sr(1) = 0.0
           sg(1) = 0.0
           sb(1) = 0.0
           sa(1) = 1.0
           
           sr(2) = 1.0
           sg(2) = 1.0
           sb(2) = 1.0
           sa(2) = 1.0
           
           goto 10
        end if
        
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

        IMPLICIT NONE
        include         'shade_vars.cmn'

	integer		beg,end,i
	character*2048	str,sfile


*******************************************************************************

*	WRITE TO MEMORY IF NO FILE NAME GIVEN
	if (end .lt. 6) then
	   mncl = ncl
           mrgb_mapping = rgb_mapping

	   do 10 i = 1,ncl
	      mslvl(i) = slvl(i)
	      msr(i) = sr(i)*100.0
 	      msg(i) = sg(i)*100.0
	      msb(i) = sb(i)*100.0
	      msa(i) = sa(i)*100.0
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
        if (rgb_mapping .eq. percent) then
           write(70,'(a/" "/"! SetPt    Red  Green   Blue  Opacity")') 
     .          'RGB_Mapping Percent'

        else if (rgb_mapping .eq. by_value) then
           write(70,'(a/" "/"! SetPt    Red  Green   Blue  Opacity")') 
     .          'RGB_Mapping By_value'

        else if (rgb_mapping .eq. by_level) then
           write(70,'(a/" "/"! Level    Red  Green   Blue  Opacity")') 
     .          'RGB_Mapping By_level'
        end if

        if (rgb_mapping .ne. by_level) then
           do i = 1,ncl
              write (70,'(5f7.1)') slvl(i),
     .             sr(i)*100.0,sg(i)*100.0,sb(i)*100.0,sa(i)*100.0
           end do
        else
           do i = 1,ncl
              write (70,'(i7,4f7.1)') i,
     .             sr(i)*100.0,sg(i)*100.0,sb(i)*100.0,sa(i)*100.0
           end do
        end if
        
	close (70)
	return

100	write (6,'(/a/)') ' PPL+ error: Unable to save spectrum ' //
     .	str(beg+9:end)//'.spk"'
101	return
	end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	SUBROUTINE SHADE_PROTECT_RESET (str)

        IMPLICIT NONE
        include         'shade_vars.cmn'
	include		'gkscm1_inc.decl'
	include		'GKSCM1.INC'
        include         'gkscm2.cmn'

	character*2048	str

*******************************************************************************

*	PROTECT PREVIOUSLY USED COLORS
	if (str .eq. 'PROTECT') then
	   protect_used_colors = .true.
	else if (str .eq. 'RESET') then
*	SET THEM TO BE REUSED
	   protect_used_colors = .false.
	   lo_color_ndx(wsid) = 0
	   hi_color_ndx(wsid) = 0
	endif

	return
	end
