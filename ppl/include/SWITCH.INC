C
C	Common block for switch data
C** 
C**    @(#)SWITCH.INC	1.1    3/10/88
C**
C
	COMMON/SWTCHC/SWITCH(20)
	COMMON/SWTCHN/SMIN(20),DEFLTS(20),FLAG(20),NFLAGS
	CHARACTER SWITCH*20
	LOGICAL DEFLTS,FLAG
	INTEGER SMIN,NFLAGS
	LOGICAL SECHO,SDEBUG,SOVER,SERASE,SWAIT,SCENTR,SRIGHT,
     *	        SLEFT,SUSER,SLATCH,SQUIET,SLOG,SYAXIS,STIME,
     *		SOFFS,SCTD,SNEXT,SVECTR
C
	EQUIVALENCE ( SECHO,FLAG( 1)),(SDEBUG,FLAG( 2)),
     *		    ( SOVER,FLAG( 3)),(SERASE,FLAG( 4)),
     *		    ( SWAIT,FLAG( 5)),(SCENTR,FLAG( 6)),
     *		    (SRIGHT,FLAG( 7)),( SLEFT,FLAG( 8)),
     *		    ( SUSER,FLAG( 9)),(SLATCH,FLAG(10)),
     *		    (SQUIET,FLAG(11)),(  SLOG,FLAG(12)),
     *		    (SYAXIS,FLAG(13)),( STIME,FLAG(14)),
     *		    ( SOFFS,FLAG(15)),(  SCTD,FLAG(16)),
     *		    ( SNEXT,FLAG(17)),(SVECTR,FLAG(18))
c
c	switch	legal qualifiers (character*20)
c	smin	minimum length for qualifier (integer)
c	deflts	default values for switches (logical)
c	flag	results of switch parsing (logical)
c	nflags	number of flags possible
c
