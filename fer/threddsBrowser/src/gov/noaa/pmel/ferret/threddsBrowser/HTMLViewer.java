/**
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
 */
package gov.noaa.pmel.ferret.threddsBrowser;

import java.io.IOException;
import java.net.URL;

import javax.swing.JEditorPane;
import javax.swing.JScrollPane;
import javax.swing.event.HyperlinkListener;

/**
 * Used to display HTML in a scrolled JEditorPane
 * @author Karl M. Smith - karl.smith (at) noaa.gov
 */
public class HTMLViewer extends JScrollPane {

	private static final long serialVersionUID = -8436110289990707935L;

	private static final String HTML_HEADER_STRING =
		"<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3.org/TR/html4/loose.dtd\">\n" +
		"<html>\n" +
		"<head>\n" +
		"<meta http-equiv=\"Content-Type\" content=\"text/html\">\n" +
		"</head>\n" +
		"<body>\n";

	private static final String HTML_FOOTER_STRING =
		"</body>\n" +
		"</html\n";

	private JEditorPane htmlEditor;

	/**
	 * Creates this JScrolledPane containing a non-editable JEditorPane
	 */
	public HTMLViewer() {
		// Editor of view HTML description of the selected dataset
		htmlEditor = new JEditorPane();
	    htmlEditor.setEditable(false);
	    setViewportView(htmlEditor);
	}

	/**
	 * Adds a hyperlinkListener to the JEditorPane component.
	 * @see JEditorPane#addHyperlinkListener(HyperlinkListener)
	 * @param hyperlinkListener the listener to add
	 */
	public void addHyperlinkListener(HyperlinkListener hyperlinkListener) {
		// Just pass this on to the JEditorPane
		htmlEditor.addHyperlinkListener(hyperlinkListener);
	}

	/**
	 * Reset the displayed page to a new empty HTML document.
	 * This will reset the style sheet to the default.
	 */
	public void clearPage() {
		htmlEditor.setContentType("text/html");
		htmlEditor.setDocument(htmlEditor.getEditorKit().createDefaultDocument());
		htmlEditor.setText(HTML_HEADER_STRING + HTML_FOOTER_STRING);
	}

	/**
	 * Displays the given string as the body of an HTML page in the current 
	 * document in the HTMLViwer.  Standard HTML header (up to and including 
	 * the <body> tag) and footer (the </body> tag and after) tags are added 
	 * to the given string before displaying. 
	 */
	public void showHTMLBodyText(String htmlBodyString) {
		htmlEditor.setText(HTML_HEADER_STRING + htmlBodyString + HTML_FOOTER_STRING);
	}

	/**
	 * Opens and displays the HTML page at the given URL.
	 * @see JEditorPane#setPage(URL)
	 * @param url URL of the page to display
	 * @throws IOException if the JEditorPane thows one
	 */
	public void setPage(URL url) throws IOException {
		// Just pass this on to the JEditorPane
		htmlEditor.setPage(url);
	}

}
