        SUBROUTINE PATTERN_SET
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
        
*       Sets the patterns for SHADE/FILL
*       J Davison JISAO/PMEL/TMAP 10.23.98
*
*       *acm* 9/02 Check that the number of levels in a pattern file isn't
*                  larger than pattern_num (see shade_vars.cmn), the number 
*                  of pattern levels allowed to be specified in a
*                  pattern file.
* v552 *acm* 3/03 Longer strings: changes for multi-line labels.

        IMPLICIT NONE
        include         'shade_vars.cmn'
        include 'cmrdl_inc.decl'            
        include 'CMRDL.INC'            
        include 'gkscm1_inc.decl'            
        include 'GKSCM1.INC'            
        include 'gkscm2.cmn'            
        include 'cmrd_inc.decl'
        include 'CMRD.INC'
        include 'gkspar.inc'

       	logical         there
        integer         beg,end,beg_eq                        
        character*2048  str                                

******************************************************************************

*	PROCESS LEVELS AS SPECIFIED
      	call upnsquish (label,str,end)
        call squish (label,1,2048)
        
	if (m .eq. 1 .or. (m .eq. 0 .and. end .eq. 0)) then
	   call pattern_levels (str,end)
	   goto 10
	end if

*	IF A PATTERN IS TO BE READ, DO THAT AND SET LEVELS
	beg = index (str,'PATTERN')
        beg_eq = index (str,'=')
	if (beg .ne. 0 .and. 
     .       (beg_eq .gt. beg .or. beg_eq .eq. 0)) then
	   call pattern_read (label,beg,end,there)
	   if (.not.there) return
	   goto 10
	end if

*	IF A PATTERN IS TO BE SAVED, DO THAT
	beg = index (str,'SAVE')
        beg_eq = index (str,'=')
	if (beg .ne. 0 .and. 
     .       (beg_eq .gt. beg .or. beg_eq .eq. 0)) then
           call pattern_save (label,beg,end)
	   goto 10
	end if

*	SET LEVELS FOR DEFAULT PATTERNS
	beg = index (str,'DEFAULT')
	if (beg .ne. 0) then
	   call pattern_default
	   goto 10
	end if

 10     return
	end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	SUBROUTINE PATTERN_LEVELS (str,end)

        IMPLICIT NONE
        include         'shade_vars.cmn'
	include		'cmrd_inc.decl'
	include		'CMRD.INC'

	include 'errmsg.parm'

* 3/99 *kob* 	move var init off of decl list  - cleaning up after jerry.
	integer		i,j,ndx,npatp1,solid
        DATA            solid/21/

        integer         beg,end,andx,status,named_ndx
        character*2048   str   
        character*100   err_msg                             

        logical    recognized

*******************************************************************************

*	NO PARAMETERS RESETS TO ZERO PATTERNS -- SOLID COLORS
	if (m .eq. 0) then 
           npat = 1
           patterns(1) = solid
           return
	end if

        ndx = int(p(1))

*	FIRST PARAMETER NEGATIVE DELETES THAT LEVEL
 	if (ndx .lt. 0) then
           andx = abs(ndx)
           if (andx .le. npat) then
              npat = npat - 1
              
              do 5 i = andx, npat
                 patterns(i) = patterns(i+1)
 5            end do
              
              if (npat .eq. 0) then
                 write (6,'(/a/a/)') 
     .         ' PPL+ note: You have removed all patterns.'
     .        ,' The pattern list will be reset to SOLID'
                 
                 npat = 1
                 patterns(1) = solid
              end if

              return
           else
              write (6,'(/a,i2,a/)') 
     .             ' PPL+ error: Pattern ',andx,
     .             ' is not available to be removed'
              return
           end if
        end if

*       NEED PROPER SET OF ARGUMENTS OTHERWISE
        if (end .eq. 0) then
           write (6,'(/a/a/)') 
     .          ' PPL+ error: The syntax to set a pattern', 
     .          ' is the sequence ORDER and then the PATTERN name'
           return
        end if

*	REJECT INDEX OUTSIDE CURRENT LIST + 1
        npatp1 = npat + 1
	if (ndx .le. 0 .or. ndx .gt. npatp1) then
           
          write (6,'(/a,i2,a/a,i2/)') 
     .     ' PPL+ error: The pattern position you specified, ', 
     .     ndx,',',' is outside the valid range of 1 to ', npatp1
           return
        end if

*	REJECT PATTERNS WITH UNRECOGNIZED NAME        
        recognized = .false.
        do 10 i = 1, 21
           beg = index (pattern_list(i),str(1:end))
           if (beg .eq. 0) goto 10

           recognized = .true.
           named_ndx = i
           goto 11
 10     CONTINUE
        if (.not. recognized) then
           write (6,'(/a/a/)') ' PPL+ error: The pattern name ' 
     .        //str(1:end)//' is not valid.',
     .        ' Use the PPL command LIST APATTERNS for available '
     .        //'pattern names'
           
           return
        end if

*	REPLACE OLDER PATTERN WITH NEW, ADD PATTERN TO TOP, IF SPECIFIED
 11     if (ndx .eq. p(1)) then
           patterns(ndx) = named_ndx
           if (ndx .eq. npat + 1) npat = npat + 1
           IF (npat .GT. pattern_num) GOTO 1000
       else
*	IF NEW, INSERT INTO PROPER LOCATION IN PATTERN LIST
           do 15 j = npat,ndx+1,-1
              patterns(j+1) = patterns(j)
 15        end do

           npat = npat + 1
           IF (npat .GT. pattern_num) GOTO 1000
           patterns(ndx+1) = named_ndx
        end if


 5000	RETURN
        
 1000	WRITE (err_msg, 110)  pattern_num 
  110   FORMAT ('Too many levels in .pat file. Max is',i4)
        CALL ERRMSG( ferr_prog_limit, status, err_msg, *5000 )
	END

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        SUBROUTINE PATTERN_DEFAULT

        IMPLICIT NONE
        include         'shade_vars.cmn'

        integer  i

******************************************************************************

*	'DEFAULT' RESETS PATTERN SEQUENCE TO DEFAULT PATTERN SEQUENCE
        npat = 20
        do 1 i = 1, npat
           patterns(i) = default_patterns(i)
 1      end do
        
        return
        end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


	SUBROUTINE PATTERN_READ (str,beg,end,there)

        IMPLICIT NONE
        include 	'tmap_errors.parm'    
	include		'cmrd_inc.decl'
	include		'CMRD.INC'
        include         'shade_vars.cmn'

	logical		there, recognized
	integer		beg,end,i,j,speclen,lnblk,status,ndx,named_ndx
	character*2048	str,sfile,instr,label,specdir,tm_inq_path

	external 	lnblk

*******************************************************************************

*	IF NO FILE NAME GIVEN, READ IN FROM MEMORY
	if (end .lt. 9) then
	   if (mnpat .eq. 0) goto 100
	   npat = mnpat

	   do 4 i = 1,npat
	      patterns(i) = mpatterns(i)
4	   end do

	   there = .true.
	   return
	endif

	sfile = tm_inq_path (str(beg+8:end),'FER_PALETTE','.pat',
     .	.false.,status)
*	type *, sfile,str(beg+8:end),status,merr_ok
	if (status .ne. merr_ok) go to 100
	there = .true.

*	OPEN THE FILE
 5      open (70,file=sfile,status='old',err=100)
*	READ IN THE PATTERNS AND SET THEM
        npat = 0
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
      	call upnsquish (label,str,end)

*	REJECT PATTERNS WITH UNRECOGNIZED NAME        
        recognized = .false.
        do 25 i = 1, 21
           beg = index (pattern_list(i),str(1:end))
           if (beg .eq. 0) goto 25

           recognized = .true.
           named_ndx = i
           goto 27
 25     CONTINUE
        if (.not. recognized) then
           write (6,'(/a/a/)') ' PPL+ error: The pattern name ' 
     .        //str(1:end)//' is not valid.',
     .        ' Use the PPL command LIST APATTERNS for available '
     .        //'pattern names'
                      
           return
        end if

 27     ndx = int(p(1))
        patterns(ndx) = named_ndx
        npat = npat + 1
	goto 10

30	close (70)
	return

*	ERROR WITH FILE ACCESS
90	close (70)
100	there = .false.
	write (6,'(/a/)') ' PPL+ error: Unable to read "' //
     .	str(beg+8:end)//'.pat" to set shade and fill patterns.'

	return

	end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	SUBROUTINE PATTERN_SAVE (str,beg,end)

        IMPLICIT NONE
        include         'shade_vars.cmn'

	integer		beg,end,i
	character*2048	str,sfile

*******************************************************************************

*	WRITE TO MEMORY IF NO FILE NAME GIVEN
	if (end .lt. 6) then
	   mnpat = npat

	   do 10 i = 1,npat
	      mpatterns(i) = patterns(i)
10	   end do
	   goto 101
	end if

	sfile = str(beg+5:end)//'.pat'

*	OPEN THE FILEOL

	open (70,file=sfile,status='unknown',form='formatted',
     .	err=100)

*	WRITE OUT THE SPECTRUM
	do i = 1,npat
	   write (70,'(i3,2x,a25)') i, pattern_list(patterns(i))
	end do

	close (70)
	return

100	write (6,'(/a/)') ' PPL+ error: Unable to save pattern list ' //
     .	str(beg+9:end)//'.pat"'
101	return
	end

