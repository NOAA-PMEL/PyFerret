	SUBROUTINE GTQUAL(LINE,ILEN,IER,IPOS)
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
C**    @(#)gtqual.F	1.2    5/26/88
C**
C**
C***********************************************************************
C**
C**		    PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C**	Subprogram called:
C**		CALL GTQUAL(LINE,ILEN,IER,IPOS)
C**
C**	Parameters:
C**		LINE -- Input line containing qualifiers (char*255)
C**		ILEN -- Length of command portion
C**		IER  -- Error code =0 no error
C**       =3 qualifier not found
C**		IPOS -- Position of bad qualifier
C**
C**	Description:
C**		GTQUAL parses the command and qualifiers and sets the
C**		appropriate flag in the switch common blocks
C**
C**	History:
C**	    Written: 
C**		2-AUG-86	by Donald W. Denbo
C**	    Modified:
C**
C**
C       Mod *JD* 3.22.91 to permit use of / in command qualifiers for unix.
C       Not a complete fix as will screw up unix paths elsewhere in PPL

#ifdef unix
	INCLUDE 'pplv11inc/SWITCH.INC'
#else
	INCLUDE 'pplv11inc:SWITCH.INC'
#endif
	CHARACTER LINE*(*)
	LOGICAL TEST
C
C	SET DEFULTS
C
	IER=0
	IDONE=LNBLK(LINE,255)
C       Mod to allow Ferret to work in Unix as is -- using "/" for
C       qualifiers -- test mode only J Davison 3.22.91  3 ifdefs commented out
C*JD*#ifdef unix
C	ILEN = INDEX(LINE,':')-1
C#else
	ilen = index(line,'/')-1
C#endif
	IF(ILEN.LE.0)ILEN=IDONE
	DO 10 I=1,NFLAGS
10	FLAG(I)=DEFLTS(I)
C
C	PARSE LINE
C
	N=0
	IS=1
C*JD*#ifdef  unix
C20	ISLSH = INDEX(LINE(IS:),':')+IS-1
C#else
20	islsh = index(line(is:),'/')+is-1
C#endif
	IF(ISLSH.GE.IS)THEN
C
C	QUALIFIER FOUND
C
C*JD*#ifdef unix
C	    IEND = INDEX(LINE(ISLSH+1:),':')+ISLSH
C#else
	    iend = index(line(islsh+1:),'/')+islsh
C#endif
	    IF(IEND.LE.ISLSH)IEND=IDONE+1
	    TEST=LINE(ISLSH+1:ISLSH+2).NE.'NO'
	    IF(TEST)THEN
		ISLSH=ISLSH+1
	    ELSE
		ISLSH=ISLSH+3
	    ENDIF
	    LLEN=IEND-ISLSH
C
C	CHECK FOR LEGAL QUALIFIER
C
	    DO 30 J=1,NFLAGS
	    IF(LLEN.LT.SMIN(J))GOTO 30
	    IF(LINE(ISLSH:IEND-1).EQ.SWITCH(J)(1:LLEN))GOTO 31
30	    CONTINUE
	    IER=3
	    IPOS=ISLSH
	    RETURN
31	    FLAG(J)=TEST
	    IS=IEND
	    GOTO 20
	ENDIF
	RETURN
	END
