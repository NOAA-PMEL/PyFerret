
/*******************************************************************************
       SaveMacroWarning.h
       This header file is included by SaveMacroWarning.c

*******************************************************************************/

#ifndef	_SAVEMACROWARNING_INCLUDED
#define	_SAVEMACROWARNING_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/DialogS.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/MessageB.h>

extern Widget	SaveMacroWarning;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_SaveMacroWarning( swidget _UxUxParent );

#endif	/* _SAVEMACROWARNING_INCLUDED */
