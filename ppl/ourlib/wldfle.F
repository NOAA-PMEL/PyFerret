	SUBROUTINE WLDFLE(DFN,FNM,EFN,ISTAT,IFLAG)
C** 
C**    @(#)wldfle.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C
C	WLDFLE uses a file template with a supplied file name
C	to create a new file name the uses the template to fill
C	in an unspecified parts.  For example, you can use .EPIC
C	as the template and whenever you do not explicitly specify
C	a file extension .EPIC will be used.
C
C	DFN -- character, the template file name
C	FNM -- character, the user supplied file name
C	EFN -- character, the new file name (output)
C	ISTAT -- INT*4, status flag =1 success
C	IFLAG -- INT*$, a flag that gives a great deal of information
C			regarding the way the substitution took place.
C
	IMPLICIT INTEGER*4 (L)
	IMPLICIT INTEGER*2 (W)

#ifdef STAR_1_SUPPORTED
	IMPLICIT integer*1 (B)
#else
	IMPLICIT BYTE (B)
#endif

	CHARACTER DFN*(*),FNM*(*),EFN*(*),RSA*81
#ifdef unix
	EFN=FNM
#else
	INCLUDE '($RMSDEF)'
	INCLUDE '($FABDEF)'
	INCLUDE '($NAMDEF)'
	INTEGER*4 MASK,ISTAT,IFLAG,SYS$PARSE
	DATA MASK/0/
C
C	DEFINE THE FAB AND NAM
C
	RECORD /FABDEF/ FAB
	RECORD /NAMDEF/ NAM
C
C	SET THE FAB
C
	FAB.FAB$B_BID=FAB$C_BID
	FAB.FAB$B_BLN=FAB$C_BLN
	FAB.FAB$L_DNA=%LOC(DFN)
	FAB.FAB$B_DNS=MIN0(LEN(DFN),127)
	FAB.FAB$L_FNA=%LOC(FNM)
	FAB.FAB$B_FNS=MIN0(LEN(FNM),127)
	EFN=' '
	FAB.FAB$L_FOP=IBSET(MASK,FAB$V_OFP)
	FAB.FAB$W_IFI=0
	FAB.FAB$L_NAM=%LOC(NAM)
C
C	SET THE NAM
C
	NAM.NAM$B_BID=NAM$C_BID
	NAM.NAM$B_BLN=NAM$C_BLN
	NAM.NAM$L_ESA=%LOC(EFN)
	NAM.NAM$B_ESS=MIN0(LEN(EFN),127)
C	nam.nam$b_nop=IBSET(MASK,NAM$V_NOCONCEAL)
	NAM.NAM$B_NOP=IBSET(MASK,NAM$V_SYNCHK)
C
C	NAM$V_SYNCHK
C
	NAM.NAM$L_RLF=0
	NAM.NAM$L_RSA=%LOC(RSA)
	NAM.NAM$B_RSL=81
	NAM.NAM$L_FNB=0
C
C	call the system parsing routine
C
	ISTAT=SYS$PARSE(FAB)
	I=NAM.NAM$B_ESL
	EFN(I+1:)=' '
	IFLAG=NAM.NAM$L_FNB
	ISEM=INDEX(EFN,';')
	L=IBITS(IFLAG,NAM$V_wild_ver,NAM$S_wild_ver)
C
C	if the version was wild carded, i.e. * then
C	append a * to the file name
C
	IF(I.EQ.ISEM.AND.L.NE.0)THEN
	    EFN(I+1:)='* '
	ENDIF
#endif
	RETURN
	END
