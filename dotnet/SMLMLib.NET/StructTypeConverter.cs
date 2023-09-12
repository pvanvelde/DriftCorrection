using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;
using System.Globalization;

using System.ComponentModel;
using System.ComponentModel.Design.Serialization;

namespace SMLMLib
{
	public class StructTypeConverter<T> : TypeConverter where T: struct
	{
		Type structType;
		public StructTypeConverter() {
			structType = typeof(T);
		}

		public override bool GetCreateInstanceSupported ( ITypeDescriptorContext context )
		{
			return true;
		}

		/// <summary>
		/// Satisfy the CreateInstance call by reading data from the propertyValues dictionary
		/// </summary>
		public override object CreateInstance (ITypeDescriptorContext context, IDictionary propertyValues )
		{
			// Get values of "Me" and "You" properties from the dictionary, and
			// create a new instance which is returned to the caller
			object inst = new T();
			PropertyInfo[] props = typeof(T).GetProperties(BindingFlags.Instance | BindingFlags.Public);

			foreach (PropertyInfo pi in props) {
				if (pi.CanWrite)
					pi.SetValue(inst, propertyValues[pi.Name], null);
			}

			return inst;
		}

		/// <summary>
		/// Does this struct expose properties?
		/// </summary>
		public override bool GetPropertiesSupported ( ITypeDescriptorContext context )
		{
			// Yes!
			return true ;
		}

		/// <summary>
		/// Return the properties of this struct
		/// </summary>
		public override PropertyDescriptorCollection GetProperties (ITypeDescriptorContext context , object value , Attribute[] attributes)
		{
			PropertyDescriptorCollection properties = TypeDescriptor.GetProperties(value, attributes);

			List<PropertyDescriptor> r = new List<PropertyDescriptor>();
			foreach (PropertyDescriptor pd in properties)
				if (!pd.IsReadOnly)
					r.Add(pd);

			return new PropertyDescriptorCollection(r.ToArray());
		}
		
		/// <summary>
		/// Check what this type can be created from
		/// </summary>
		/// <param name="context"></param>
		/// <param name="sourceType"></param>
		/// <returns></returns>
		public override bool CanConvertFrom (ITypeDescriptorContext context, System.Type sourceType)
		{
			// Just strings for now
//			bool canConvert = (sourceType == typeof(string));

	//		if (!canConvert)
		bool		canConvert = base.CanConvertFrom(context, sourceType);

			return canConvert;
		}

		public override object ConvertFrom(System.ComponentModel.ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value)
		{
			object retVal = null;
			/*
			string sValue = value as string;
			
			if (sValue != null)
			{
				// Check that the string actually has something in it...
				sValue = sValue.Trim();

				if (sValue.Length != 0)
				{
					// Parse the string
					if (null == culture)
						culture = CultureInfo.CurrentCulture;

					// Split the string based on the cultures list separator
					string[] parms = sValue.Split(';');

					if (parms.Length == 3)
					{
						// Should have an integer and a string.
						int me = Convert.ToInt32(parms[0]);
						string you = parms[1];

						// And finally create the object
						retVal = new Doofer(me, you);
					}
				}
			}
			else */
				retVal = base.ConvertFrom(context, culture, value);
			
			return retVal;
		}

		public override bool CanConvertTo (ITypeDescriptorContext context, System.Type destinationType)
		{
			// InstanceDescriptor is used in the code behind
			bool canConvert = (destinationType == typeof(InstanceDescriptor));

			if (!canConvert)
				canConvert = base.CanConvertFrom(context, destinationType);

			return canConvert;
		}
		
		public override object ConvertTo(System.ComponentModel.ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value, System.Type destinationType)
		{
			object retVal = null;

/*			return null;
			
			// If this is an instance descriptor...
			if (destinationType == typeof(InstanceDescriptor))
			{
				FieldInfo[] fields = typeof(T).GetFields();
				
				Type[] argTypes = new System.Type[fields.Length];

				for (int i = 0; i < fields.Length; i++)
					argTypes[i] = fields[i].FieldType;

				// Lookup the appropriate Doofer constructor
				ConstructorInfo constructor = typeof(T).GetConstructor(argTypes);

				object[] arguments = new object[2];

				arguments[0] = doofer.Me;
				arguments[1] = doofer.You;

				// And return an instance descriptor to the caller. Will fill in the CodeBehind stuff in VS.Net
				retVal = new InstanceDescriptor(constructor, arguments);
			}
			else if (destinationType == typeof(string))
			{
				// If it's a string, return one to the caller
				if (null == culture)
					culture = CultureInfo.CurrentCulture;

				string[] values = new string[2];

				// I'm a bit of a culture vulture - do it properly!
				TypeConverter numberConverter = TypeDescriptor.GetConverter(typeof(int));

				values[0] = numberConverter.ConvertToString(context, culture, doofer.Me);
				values[1] = doofer.You;

				// A useful method - join an array of strings using a separator, in this instance the culture specific one
				retVal = String.Join(culture.TextInfo.ListSeparator + " ", values);
			}
			else*/
				retVal = base.ConvertTo(context, culture, value, destinationType);

			return retVal;
		}
	}
}
