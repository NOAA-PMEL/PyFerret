	PARAMETER (IBFDIM=10)
C** 
C**    @(#)CMDCOM.INC	1.1    3/10/88
C**
C
C	Common block for command line input, prompt and line
C	count for both terminal and file.
C
	COMMON/CMDLNC/CMDFIL,PRMPT,CMFILE(8),CMDBUF(IBFDIM)
	COMMON/CMDLNN/CMDLUN,INP,CMDLIN,ECHLUN,CMDLEV,LINCNT,
     *		      CMLINE(0:8),MBUFF,BUFSZE,CMECHO(0:8),CMDEBG(0:8),
     *		      CMQUIE(0:8),CMLOG(0:8),IFLEV,WHLEV,WHLINE(0:8),
     *		      CMIFLV(0:8),CMWHLV(0:8),CMWHLN(0:8,8)
	CHARACTER CMDFIL*80,PRMPT*30,CMFILE*80,CMDBUF*255
	INTEGER CMDLUN,CMDLIN,ECHLUN,CMDLEV,LINCNT,CMLINE,
     *		      BUFSZE,IFLEV,WHLEV,WHLINE,CMIFLV,CMWHLV,CMWHLN
	LOGICAL CMECHO,CMDEBG,CMQUIE,CMLOG
c
c	cmdfil	current command file name (char*80)
c	prmpt	current interactive prompt (char*30)
c	cmfile(8) storage of command file names up to 8 levels deep
c	cmdbuf()  storage for commands to be executed
c
c	cmdlun	logical unit to read commands from (integer)
c	inp	length of prompt in characters
c	cmdlin	unused
c	echlun	echo file logical unit (integer)
c	cmdlev	current level of nesting for command files
c	lincnt  curent position in current command file
c	cmline(8) storage of line positions for 8 levels of command
c	mbuff	next command to be read from cmdbuf
c	bufsze	number of commands in cmdbuf
c	cmecho	storage for old echo state
c	cmdebg	storage for old debug state
c	cmquie	storage for old quiet state
c	cmlog	storage for old log state
c	iflev	current if level count
c	whlev	current while level count
c	whline	current while return line numbers
c	cmiflv	storage for if level count
c	cmwhlv	storage for while level count
c	cmwhln	storage for while return line numbers
