using SMLMLib;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TiffViewer
{
	public struct LocalizationResult
	{
		public Vector3 pos;
		
	}

	interface ILocalizationMethod
	{
		object GetConfig();
		ImageQueue CreateImageQueue(Size imageSize, object config);
		string DisplayName { get; }
	}

}
