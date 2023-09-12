/*
 * To the extent possible under law, the ImageJ developers have waived
 * all copyright and related or neighboring rights to this tutorial code.
 *
 * See the CC0 1.0 Universal license for details:
 *     http://creativecommons.org/publicdomain/zero/1.0/
 */

package com.photonj.imagej;

import net.imagej.DatasetService;

import javax.swing.SwingUtilities;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;

import org.scijava.app.StatusService;
import org.scijava.command.Command;
import org.scijava.command.CommandService;
import org.scijava.display.DisplayService;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.thread.ThreadService;
import org.scijava.ui.UIService;

import io.scif.services.DatasetIOService;



@Plugin(type = Command.class, menuPath = "Plugins>PhotonJ UI")
public class PhotonJ implements Command {

	@Parameter
	OpService ops;

	@Parameter
	LogService log;

	@Parameter
	UIService ui;

	@Parameter
	CommandService cmd;

	@Parameter
	StatusService status;

	@Parameter
	ThreadService thread;

	@Parameter
	DisplayService display;
	
	@Parameter
	private DatasetService datasetService;

	@Parameter
	private DatasetIOService datasetIOService;

	
	private static MainUI dialog = null;

    @Override
    public void run() {
    	    	
		SwingUtilities.invokeLater(() -> {
			if (dialog == null) {
				dialog = new MainUI();
			}
			dialog.setVisible(true);
			dialog.setUi(ui);			
		});
    }

    /**
     * This main function serves for development purposes.
     * It allows you to run the plugin immediately out of
     * your integrated development environment (IDE).
     *
     * @param args whatever, it's ignored
     * @throws Exception
     */
    public static void main(final String... args) throws Exception {
        // create the ImageJ application context with all available services
    	
    	//ZStack s = ZStack.readFromFile("C:\\dev\\photonpy-me\\projects\\csplines\\psfsim.zstack");
    	
        final ImageJ ij = new ImageJ();
        ij.ui().showUI();
    }

}
