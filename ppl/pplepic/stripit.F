        subroutine stripit (instring,inlength,cchar,outstring,il)
C** 
C**    @(#)stripit.f	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Removes character cchar from instring(1:inlength).
c
c  Input:
c       instring = input string
c       inlength = length of input string
c       cchar = character to remove from instring
c  Output:
c       outstring = output string, equal to input string with character
c                   cchar removed remaining string compressed.
c       il = length of output string
c
        character*(*) instring,outstring
        character*1 cchar
c
        outstring=instring
        il=inlength
c
c  Get rid of "cchar"
c
200         i=index(outstring(1:il),cchar)
            if (i.eq.0) goto 300
            if (i .eq. 1) then
                outstring(1:il)=outstring(i+1:il)
                il=il-1
            else
                outstring(1:il)=outstring(1:i-1)//outstring(i+1:il)
                il=il-1
            endif
            goto 200
300     return
        end
