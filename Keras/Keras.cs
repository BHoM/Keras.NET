using Keras.Layers;
using Keras.Utils;
using Numpy;
using Numpy.Models;
using Python.Included;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;
//using static Python.Runtime.Py;

namespace Keras
{
    public static class Keras
    {
        /***************************************************/
        /**** Public Properties                         ****/
        /***************************************************/

        //public static PyObject Instance { get { return _instance.Value; } }

        public static dynamic keras { get { return _instance.Value; } private set { keras = value; } }

        public static dynamic tensorflow { get; set; } = null;

        //public static dynamic keras2onnx { get; set; } = null;

        //public static dynamic tfjs { get; set; } = null;


        /***************************************************/
        /**** Private Fields                            ****/
        /***************************************************/

        private static Lazy<PyObject> _instance = new Lazy<PyObject>(() =>
        {
            Installer.SetupPython().Wait();
            PythonEngine.Initialize();
            return GetModule("tensorflow.keras");
        });


        /***************************************************/
        /**** Public Methods                            ****/
        /***************************************************/

        public static PyObject ToPython(object obj)
        {
            if (obj == null) return Runtime.GetPyNone();
            switch (obj)
            {
                // basic types
                case int o: return new PyInt(o);
                case float o: return new PyFloat(o);
                case double o: return new PyFloat(o);
                case string o: return new PyString(o);
                case bool o:
                    if (o)
                        return new PyObject(Runtime.PyTrue);
                    else
                        return new PyObject(Runtime.PyFalse);

                // sequence types
                case Array o: return ToList(o);
                // special types from 'ToPythonConversions'
                case Shape o: return ToTuple(o.Dimensions);
                case ValueTuple<int> o: return ToTuple(o);
                case ValueTuple<int, int> o: return ToTuple(o);
                case ValueTuple<int, int, int> o: return ToTuple(o);
                case Slice o: return o.ToPython();
                case PythonObject o: return o.PyObject;
                case PyObject o: return o;
                case Sequence o: return o.PyInstance;
                case StringOrInstance o: return o.PyObject;
                case KerasFunction o: return o.PyObject;
                case Base o: return o.PyInstance;
                default: throw new NotImplementedException($"Type is not yet supported: { obj.GetType().Name}. Add it to 'ToPythonConversions'");
            }
        }

        /***************************************************/

        public static PyTuple ToTuple(Array input)
        {
            var array = new PyObject[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                array[i] = ToPython(input.GetValue(i));
            }

            return new PyTuple(array);
        }

        /***************************************************/

        public static PyTuple ToTuple(ValueTuple<int> input)
        {
            var array = new PyObject[1];
            array[0] = ToPython(input.Item1);

            return new PyTuple(array);
        }

        /***************************************************/

        public static PyTuple ToTuple(ValueTuple<int, int> input)
        {
            var array = new PyObject[2];
            array[0] = ToPython(input.Item1);
            array[1] = ToPython(input.Item2);

            return new PyTuple(array);
        }

        /***************************************************/

        public static PyTuple ToTuple(ValueTuple<int, int, int> input)
        {
            var array = new PyObject[3];
            array[0] = ToPython(input.Item1);
            array[1] = ToPython(input.Item2);
            array[2] = ToPython(input.Item3);

            return new PyTuple(array);
        }

        /***************************************************/

        public static PyList ToList(Array input)
        {
            var array = new PyObject[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                array[i] = ToPython(input.GetValue(i));
            }

            return new PyList(array);
        }

        /***************************************************/
        /**** Private Methods                           ****/
        /***************************************************/

        private static PyObject GetModule(string name)
        {
            TryInstall(name);
            return Py.Import(name);
        }

        /***************************************************/

        private static void SetModules()
        {
            if (TryInstall("tensorflow"))
                tensorflow = Py.Import("tensorflow");

            if (TryInstall("keras"))
                keras = Py.Import("keras");

            //if (TryInstall("onnxmltools"))
            //    tensorflow = Py.Import("onnxmltools");

            //if (TryInstall("tensorflowjs"))
            //    tensorflow = Py.Import("tensorflowjs");
        }

        /***************************************************/

        private static bool TryInstall(string module)
        {
            try
            {
                if (!Python.Included.Installer.IsModuleInstalled(module))
                    Python.Included.Installer.PipInstallModule(module);
                Py.Import(module);

                return true;
            }
            catch
            {
                return false;
            }
        }

        /***************************************************/
    }
}