/* JC_CallbackUtility.h
 *
 * Jonathan Callahan
 * Feb 19th 1996
 *
 * This file contains utility functions for common tasks.
 *
 * I have endeavored to stay very close to true Object-Oriented-Programming
 * principles and have in general followed the C++ guidelines as I learned
 * them from "Weiskamp & Flamig:  The complete C++ primer".
 *
 */


#ifndef _JC_CALLBACKUTILITY_H
#define _JC_CALLBACKUTILITY_H

#include <ctype.h>
#include <stdio.h>
#include <Xm/Xm.h>
 

/* .................... JC_String methods .................... */

extern void JC_TextField_VerifyNumeric_CB( Widget wid, XtPointer client_data, XtPointer call_data );

/*
 * Input     wid:
 *           client_data:
 *           call_data:    pointer to an XmTextVerifyCallbackStruct
 *
 * Output    Will not allow the user to enter a non-digit/non-period into a textField.  
 */


extern void JC_Message_CB( Widget wid, XtPointer client_data, XtPointer call_data );

/*
 * Input     wid:
 *           client_data:  message to be printed
 *           call_data:    UNUSED
 *
 * Output    Creates a message dialog using "client_data" as the text string.
 *
 */




#endif /* _JC_CALLBACKUTILITY_H */

/* ~~~~~~~~~~~~~~~~~~~~ END OF JC_CommandGen.h ~~~~~~~~~~~~~~~~~~~~ */
