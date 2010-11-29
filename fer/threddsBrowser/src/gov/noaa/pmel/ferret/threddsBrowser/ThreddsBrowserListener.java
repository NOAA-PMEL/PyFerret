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

import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Point;
import java.awt.Window;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.net.Authenticator;

import javax.swing.JFrame;
import javax.swing.JOptionPane;

import opendap.dap.DConnect2;

import org.apache.commons.httpclient.HttpClient;

import thredds.ui.UrlAuthenticatorDialog;

import ucar.nc2.dataset.NetcdfDataset;
import ucar.nc2.stream.CdmRemote;
import ucar.nc2.ui.WmsViewer;
import ucar.nc2.util.net.HttpClientManager;
import ucar.unidata.io.http.HTTPRandomAccessFile;
import ucar.util.prefs.PreferencesExt;
import ucar.util.prefs.XMLStore;

/**
 * A listener for a ThreddsBrowser as well as the Window containing that Browser.
 * @author Karl M. Smith - karl.smith (at) noaa.gov
 */
public class ThreddsBrowserListener extends WindowAdapter implements ActionListener {

	/** XMLStore key for a Point giving the location of the main Window given to the browser */
	public static final String WINDOW_LOCATION = "WindowLocation";

	/** Window containing the ThreddsBrowser to listener for WindowClosing events */
	private Window mainWindow;

	/** ThreddsBrowser to listen to for Action events */
	private ThreddsBrowser tBrowser;

	/** Whether or not to use the selected dataset name in the ThreddsBrowser when an Action event occurs */ 
	private boolean useDataset;

	/** XMLStore to save the Window Location before closing the Window from either Action or WindowClosing events */
	private XMLStore givenStore;

	/**
	 * Create a Listener for Action events from a ThreddsBrowser and WindowClosing events from the
	 * Window containing that ThreddsBrowser.  
	 * <ul>
	 * <li>Action event if useDataset is true: the dataset name is retrieved from tBrowser and 
	 * printed as a "USE ..." String to System.out, and the windowClosing method is then called.
	 * If no dataset name is selected in tBrowser, an error dialog is posted and no further action 
	 * is taken.  
	 * <li>Action event if useDataset is false: the windowClosing method is just called.
	 * <li>WindowClosing events (and when the windowClosing method is called from Action events): 
	 * the current location of mainWindow as well as the setting of tBrowser are saved to the file 
	 * that is backing givenStore, the window is closed, and the 
	 * application exits.
	 * <ul>
	 * @param mainWindow the Window containing tBrowser.
	 * @param tBrowser the ThreddsBrowser to use.
	 * @param useDataset whether to use the selected dataset from tBrowser.
	 * @param givenStore the XMLStore used to save current settings.
	 */
	public ThreddsBrowserListener(Window mainWindow, ThreddsBrowser tBrowser, boolean useDataset,  XMLStore givenStore) {
		if ( mainWindow == null )
			throw new NullPointerException("null Window given to the ThreddsBrowserListener constructor");
		if ( tBrowser == null )
			throw new NullPointerException("null ThreddsBrowser given to the ThreddsBrowserListener constructor");
		if ( givenStore == null )
			throw new NullPointerException("null XMLStore given to the ThreddsBrowserListener constructor");
		this.mainWindow = mainWindow;
		this.tBrowser = tBrowser;
		this.useDataset = useDataset;
		this.givenStore = givenStore;
	}

	/**
	 * If useDataset was true in the creation of this Listener, the dataset name is retrieved
	 * from the ThreddsBrowser and printed as a "USE ..." to System.out and the application exits.
	 * If no dataset name is selected in the ThreddsBrowser, an error dialog is posted and no 
	 * further action is taken.
	 * If useDataset was false, the application just exits.
	 */
	@Override
	public void actionPerformed(ActionEvent e) {
		if ( useDataset ) {
			String datasetName = tBrowser.getDatasetName();
			if ( datasetName == null ) {
				JOptionPane.showMessageDialog(tBrowser, "No dataset is selected",
											  "No Dataset Selected", JOptionPane.ERROR_MESSAGE);
				return;
			}
			System.out.println("USE \"" + datasetName + "\"");
			System.out.flush();
		}
		windowClosing(null);
	}

	/**
	 * Saves the current location of the Window as well as the setting of the ThreddsBrowser
	 * to the XMLStore used in the creation of this Listener.
	 */
	@Override
	public void windowClosing(WindowEvent e) {
		PreferencesExt prefs = givenStore.getPreferences();

		// Save the location of main window
        Point location = mainWindow.getLocation();
        prefs.putBeanObject(WINDOW_LOCATION, location);

        // Save the ThreddsBrowser's preferences
		tBrowser.savePreferences(prefs);

		// Save the store object to the original file
		try {
			givenStore.save();
		} catch ( Exception exc ) {
			; // don't care
		}

		// Close the window and exit the application
		mainWindow.setVisible(false);
		mainWindow.dispose();
		System.exit(0);
	}

	/**
	 * Creates and displays the ThreddsBrowser as well as initializes the 
	 * HTTP client.  Any arguments given are passed on to the created
	 * ThreddsBrowser selectLocation method.
	 */
	public static void createAndShowBrowser(final String[] args) {
		// Create the window - leave the default close operation (HIDE_ON_CLOSE) 
		// since we are providing a windowClosing event listener that calls System.exit
		JFrame mainFrame = new JFrame("THREDDS Catalog Browser");
		mainFrame.setLayout(new GridBagLayout());

		// Get the default givenStore filename
		String storeFilename = ThreddsBrowser.createDefaultStoreFilename();

		// Open the givenStore of preferences
		XMLStore store;
		try {
	    	store = XMLStore.createFromFile(storeFilename, null);
	    } catch (Exception e) {
	    	// Create a read-only empty givenStore
			store = new XMLStore();
	    }
	    PreferencesExt prefs = store.getPreferences();

	    // Create the ThreddsBrower and add it to mainFrame
	    ThreddsBrowser tBrowser = new ThreddsBrowser(prefs, "FER_DATA_THREDDS");
	    GridBagConstraints gbc = new GridBagConstraints();
	    gbc.anchor = GridBagConstraints.NORTHWEST;
	    gbc.gridx = 0;     gbc.gridy = 0;
	    gbc.gridwidth = 1; gbc.gridheight = 1;
	    gbc.weightx = 1.0; gbc.weighty = 1.0;
	    gbc.fill = GridBagConstraints.BOTH;
	    mainFrame.add(tBrowser, gbc);

	    // Show the window
		mainFrame.pack();
		mainFrame.setVisible(true);

		// Set the location of the main window
		Point location = (Point) prefs.getBean(WINDOW_LOCATION, null);
		if ( location == null ) {
			location = new Point(50, 50);
		}
	    mainFrame.setLocation(location);

	    // Add a listener for the Use Action events
	    tBrowser.addUseActionListener(new ThreddsBrowserListener(mainFrame, tBrowser, true, store));

	    // Add a listener for Cancel Action events and WindowClosing events 
	    ThreddsBrowserListener cancelListener = new ThreddsBrowserListener(mainFrame, tBrowser, false, store);
	    tBrowser.addCancelActionListener(cancelListener);
		mainFrame.addWindowListener(cancelListener);

		// Initialize the URL authenticator and HTTP client manager
	    UrlAuthenticatorDialog provider = new UrlAuthenticatorDialog(mainFrame);
	    Authenticator.setDefault(provider);
	    HttpClient client = HttpClientManager.init(provider, "ThreddsBrowser");
	    DConnect2.setHttpClient(client);
	    HTTPRandomAccessFile.setHttpClient(client);
	    CdmRemote.setHttpClient(client);
	    NetcdfDataset.setHttpClient(client);
	    WmsViewer.setHttpClient(client);

	    // Pass the arguments on to the ThreddsBrowser's selectLocation method.
	    // This brings up a LocationSelectorDialog if no arguments are given.
	    tBrowser.selectLocation(args);
	}

}
