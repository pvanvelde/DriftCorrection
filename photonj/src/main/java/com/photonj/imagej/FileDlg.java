package com.photonj.imagej;

import java.io.File;

import javax.swing.JFileChooser;
import javax.swing.filechooser.FileFilter;

public class FileDlg {

	public static class Filter extends FileFilter {	
	    String desc,ext;
	
	    public Filter(String ext,String desc) {
	    	this.desc=desc;
	    	this.ext=ext;
	    }
	
	    @Override
	    public boolean accept(File f) {
	        if (f.isDirectory())
	            return true;
	        
	        if (ext.length() > 0)
	        	return (f.getName().toLowerCase().endsWith(ext));
	        return true;
	    }
	
	    @Override
	    public String getDescription() {
	        return desc;
	    }
	}

	public static String saveDialog(String title, String filters, String directory) {
		JFileChooser fc = new JFileChooser();
		addFilters(fc,filters);

		int ret = fc.showSaveDialog(null);
		if (ret == JFileChooser.APPROVE_OPTION) {
		     return fc.getSelectedFile().getPath();
		}
		return null;
	}

	private static void addFilters(JFileChooser fc, String filters) {
		String[] filterList = filters.split("&");
		for (int i=0;i<filterList.length;i++) {
			String[] s = filterList[i].split(";");
			fc.addChoosableFileFilter(new Filter(s[1],s[0]));
		}
	}
	
	
}
