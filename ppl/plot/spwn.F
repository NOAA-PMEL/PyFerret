	SUBROUTINE SPWN
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
C**    @(#)spwn.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C** v552 *acm* 3/03 Longer strings: changes for multi-line labels

#ifdef unix
	INCLUDE 'CMRDL.INC'
	INCLUDE 'LUNITS.INC'
	INCLUDE 'SYSTEM.INC'
C
	IF(.NOT.QUIETF)WRITE(LTTOUT,998)ICOM
998	FORMAT(' COMMAND : ',A12,' NOT RECOGNIZED')
c
#else
        INCLUDE 'pplinc:CMRDL.INC'
        INCLUDE 'pplinc:LUNITS.INC'
        INCLUDE 'CMDINC:SYSTEM.INC'
C
        CHARACTER*20 FILENAME(5),WILDFILE*70,WHOLENAME*70        
        INTEGER*4 STATUS,CONTEXT,LBR$OUTPUT_HELP,LIB$FIND_FILE
        INTEGER*4 LIB$SPAWN,SYS$EXIT,IFLAG,HLPFLG
        EXTERNAL LIB$GET_INPUT,LIB$PUT_OUTPUT
        INCLUDE '($RMSDEF)'
        INCLUDE '($HLPDEF)'
        IF(ICOM.EQ.'HLP')THEN
            IF(QUIETF)RETURN
            HLPFLG=HLP$M_PROCESS
            WHOLENAME='PPLUS '//LABEL
            STATUS=LBR$OUTPUT_HELP(LIB$PUT_OUTPUT,,WHOLENAME,
     *      'SYS$HELP:HELPLIB',HLPFLG,LIB$GET_INPUT)
        ELSE IF(ICOM.EQ.'HELP')THEN
            IF(QUIETF)RETURN
            WRITE(LTTOUT,'('' TYPE "HLP" FOR PLOT+ HELP'')')
            HLPFLG=HLP$M_PROCESS .OR. HLP$M_PROMPT .OR. HLP$M_LIBLIST
            STATUS=LBR$OUTPUT_HELP(LIB$PUT_OUTPUT,,LABEL,
     *      'SYS$HELP:HELPLIB',HLPFLG,LIB$GET_INPUT)
        ELSE IF(ICOM.EQ.'SPAWN')THEN
            WRITE(LTTOUT,'('' type "LOGOUT" to return to PLOT+'')')
            STATUS=LIB$SPAWN()
        ELSE IF(ICOM.EQ.'DIR')THEN
            IF(QUIETF)RETURN
            ILEN=LNBLK(LABEL,2048)
            CALL WLDFLE('*.*;*',LABEL(:ILEN),WILDFILE,STATUS,IFLAG)
            WRITE(LTTOUT,999)WILDFILE
999         FORMAT(/' SEARCH : ',A67/)
            CONTEXT=0
c
100         DO 200 ICNT = 1,4
            STATUS = LIB$FIND_FILE(WILDFILE,WHOLENAME,CONTEXT)
            IF (.NOT. STATUS) GOTO 300
            FILENAME(ICNT) = 
     >      WHOLENAME(INDEX(WHOLENAME,']')+1:INDEX(WHOLENAME,' ')-1)
200         CONTINUE
            WRITE(LTTOUT,1030) (FILENAME(I)(1:17),I=1,4)
1030        FORMAT(1X,A17,3A20)
            GOTO 100
c
300         IF (STATUS .EQ. RMS$_NMF) THEN
                WRITE(LTTOUT,1030) (FILENAME(I)(1:17),I=1,ICNT-1)
            ELSE IF (STATUS .EQ. RMS$_SYN) THEN
                WRITE(LTTOUT,1040) LABEL
1040            FORMAT(' ERROR IN DIRECTORY SPECIFICATION ::'/1X,A70/
     >          ' correct example : [TEST.SUB.SUB2]')
            ELSE IF (STATUS .EQ. RMS$_DNF) THEN
                WRITE(LTTOUT,1050) LABEL
1050            FORMAT(' ERROR - DIRECTORY NOT FOUND ::'/1X,A70)
            ELSE IF (STATUS .EQ. RMS$_FNF) THEN
                WRITE(LTTOUT,1055)LABEL
1055            FORMAT(' FILE NOT FOUND :: ',A50)
            ELSE
                WRITE(LTTOUT,1060)
1060            FORMAT(' SYSTEM ERROR :')
                CALL SYS$EXIT(%VAL(STATUS))
            ENDIF
        ELSE
                IF(.NOT.QUIETF)WRITE(LTTOUT,998)ICOM
998             FORMAT(' COMMAND : ',A12,' NOT RECOGNIZED')
        ENDIF
#endif
	RETURN
	END
