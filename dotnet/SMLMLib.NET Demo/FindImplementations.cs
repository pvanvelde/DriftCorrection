using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TiffViewer
{
	public static class FindImplementations
	{
		public static Type[] Find(Type interfaceType)
		{
			var types = AppDomain.CurrentDomain.GetAssemblies()
				.SelectMany(s => s.GetTypes())
				.Where(p => !p.IsAbstract && !p.IsInterface && p.GetConstructor(Type.EmptyTypes) != null && interfaceType.IsAssignableFrom(p));

			return types.ToArray();
		}
	}
}
