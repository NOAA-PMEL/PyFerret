C
C	Common block for system flags etc...
C** 
C**    @(#)SYSTEM.INC	1.1    3/10/88
C**
C
	COMMON/STATUS/ECHOF,DEBUGF,BATCHF,TERMF,ERRORF,SMKEY2,
     *	DONEF,MEMBUF,LOGITF,QUIETF,LOGCMF,SKIPIF,SKIPWH
	LOGICAL ECHOF,DEBUGF,BATCHF,TERMF,ERRORF,SMKEY2,DONEF,
     *	MEMBUF,LOGITF,QUIETF,LOGCMF,SKIPIF,SKIPWH
c
c	echof	echo file output switch
c	debugf	debug output mode
c	batchf	batch mode (rsx only)
c	termf	running from a terminal
c	errorf	error found (unused)
c	smkey2	reading from user keyfile
c	donef	atend signals return to non-existant level
c		i.e., return to original calling program without
c		further action
c	membuf	use STRING for input (GETLIN)
c	logitf	logging of program use flag
c	quietf	no messages to terminal if true
c	logcmf	log commands to terminal from command file if true
c	skipif	if .true. skip command lines  until else of endif
c	skipwh	if .true. skip command lines until endw
c
