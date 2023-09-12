package com.photonj.imagej;

import javax.swing.JCheckBox;
import javax.swing.JComponent;
import javax.swing.JRadioButton;
import javax.swing.JSlider;
import javax.swing.JSpinner;
import javax.swing.JTextField;
import javax.swing.SpinnerNumberModel;

import ij.Prefs;

public class PersistUIState {
	
	public static String getKey(JComponent c) {
		
		if (c.getName() == null)
			throw new RuntimeException("PersistUIState used on component without name");
		
		return "photonj." + c.getName();
	}

	public static void loadState(JComponent[] comp) {
		
		for (int i=0;i<comp.length;i++) {
			JComponent c = comp[i];
			String key = getKey(c);
			
			// for some reason all these imagej parsing methods (getDouble, getInt, getBoolean) dont work and all return their default value.
			// String values are the only ones that work.
			String val = Prefs.get(key, null);
			if (val == null)
				continue;

			System.out.println("Loading state for UI component "+ c.getName() + ", value=" + val);
			if (c instanceof JSpinner) {
				JSpinner s = (JSpinner)c;
				
				if (s.getValue().getClass() == Float.class) {
					//double v = Prefs.getDouble(key, -1.0);
					s.setValue(Float.parseFloat(val));
				} else if (s.getValue().getClass() == Integer.class) {
					s.setValue(Integer.parseInt(val));
				}
				else {
					throw new RuntimeException("Can't set value of " + c.getName());
				}
				
			}
			else if (c instanceof JSlider)
				((JSlider)c).setValue(Integer.parseInt(val));
			else if (c instanceof JTextField) {
				JTextField tf = (JTextField)c;
				tf.setText(val.trim());
			}
			else if (c instanceof JCheckBox) {
				((JCheckBox)c).setSelected(Boolean.parseBoolean(val));
			} else if(c instanceof JRadioButton) {
				((JRadioButton)c).setSelected(Boolean.parseBoolean(val));
			} else
				throw new RuntimeException("Unknown type " + c.getClass().getName() + " for UI component " + c.getName());
		}
	}
	
	public static void storeState(JComponent[] comp) {
		System.out.println("Saving PhotonJ settings..");
		
		for (int i=0;i<comp.length;i++) {
			JComponent c = comp[i];
			String key = getKey(c);

			if (c instanceof JSpinner)  {
				JSpinner s = (JSpinner)c;
				Object o = s.getValue();
				if (o instanceof Integer)
					Prefs.set(key, (int)o);
				else if (o instanceof Double)
					Prefs.set(key, ((Double) o).floatValue());
				else if (o instanceof Float)
					Prefs.set(key, ((Float)o).floatValue());
				else
					throw new RuntimeException("Component " + c.getName() + " has unimplemented value type");
			}
			else if (c instanceof JSlider)
				Prefs.set(key, ((JSlider)c).getValue());
			else if (c instanceof JTextField)
				Prefs.set(key, ((JTextField)c).getText());
			else if (c instanceof JCheckBox)
				Prefs.set(key, ((JCheckBox)c).isSelected());
			else if(c instanceof JRadioButton)
				Prefs.set(key, ((JRadioButton)c).isSelected());
			else
				throw new RuntimeException("Unknown type " + c.getClass().getName() + " for UI component " + c.getName());
			
			String val = Prefs.get(key, null);
			System.out.println("Set " + key + " = " + val);
		}
		
	}
}
