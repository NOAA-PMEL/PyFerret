	INTEGER FUNCTION HASH_NAME( name, npiles )

* produce a pseudo-random number between 1 and npiles from the given name

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/7/87

* calling argument declarations
	INTEGER		npiles
	CHARACTER*(*)	name

* local variable declarations
	INTEGER	TM_LENSTR1, len_name, pile, i, new_char

	len_name = TM_LENSTR1( name )

* generate pseudo-random integer based on name
	pile = 0
	DO 100 i = 1, len_name
	   new_char = MOD( 2107*ICHAR(name(i:i)), 531 )
	   pile = IEOR( pile, new_char )
 100	CONTINUE

* modulo to number of piles indicated
	HASH_NAME = MOD( pile, npiles ) + 1

	RETURN
	END
