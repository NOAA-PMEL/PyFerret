	SUBROUTINE TM_Parse_String ( instr, ndim, outary, nvals )
*
* Will take the input string "instr" and separate the individual items at space,
* tab, or commas. It will place the items in array outary, which is dimensioned
* ndim, and then return number of items found in nvals. If more than ndim items
* are found, then only ndim items will be saved, although total number of items
* found will be reported in nvals.
*
* revision 0.00 - 08/14/89
*
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.

* Arguments:
*	instr	- input string containing items to be parsed
*	ndim	- dimension of outary
*	outary	- array where individual items are stored
*	nvals	- number of items found in instr
*
	INTEGER*4	ndim, nvals
	CHARACTER*(*)	instr, outary(ndim)
*
* Local definitions
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
	CHARACTER*1	blank, comma, tab
	PARAMETER      (blank = ' ',
     .                  comma = ',',
     .                	tab = 	'	')
	INTEGER*4	ilen, pos, start
*
* Initialize
	nvals = 0
	ilen = LEN(instr)
	pos = 1
*
* Locate non-blank, non-comma, non-tab
  100	IF (pos .GT. ilen) GOTO 900
	IF ( instr(pos:pos) .NE. blank .AND.
     .	     instr(pos:pos) .NE. comma .AND.
     .	     instr(pos:pos) .NE. tab         ) THEN
	  start = pos
	  GOTO 200
	ENDIF
	pos = pos + 1
	GOTO 100
*
* Found start of item, find blank, comma, or tab to locate end of item
  200	IF ( instr(pos:pos) .EQ. blank .OR.
     .	     instr(pos:pos) .EQ. comma .OR.
     .	     instr(pos:pos) .EQ. tab         ) GOTO 300
	pos = pos + 1
	IF (pos .GT. ilen) GOTO 300
	GOTO 200
*
* Found end of item, store it.
  300	nvals = nvals + 1
	IF (nvals .LE. ndim) outary(nvals) = instr(start:pos-1)
	GOTO 100
*
* End of string
  900	RETURN
	END
